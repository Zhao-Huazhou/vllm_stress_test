import time
import requests
import concurrent.futures
import random   
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/mnt/data2/stress_test/DeepSeek-R1-AWQ/")

import json
########################
# 配置参数
########################
URL = "http://0.0.0.0:8080/v1/chat/completions"#vllm模型接口地址
MODEL = "deepseek-r1"#vllm模型名称

MAX_CONCURRENT = 512
TIMEOUT_THRESHOLD = 180.0
ERROR_THRESHOLD = 0.9
CONCURRENCY_STEP = 2  # 并发递增步长                          # 允许的最大错误率，超过该错误率后测试结束
# 添加 JSON 文件路径和读取逻辑
QUERY_FILE = "./zh-data-part-00.json"  # JSON 文件路径
try:
    with open(QUERY_FILE, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    # QUERY = queries.get('query', '')  # 从 JSON 中获取 query 字段
except FileNotFoundError:
    print(f"错误: 找不到文件 {QUERY_FILE}")
    exit(1)
except json.JSONDecodeError:
    print(f"错误: {QUERY_FILE} 不是有效的 JSON 文件")
    exit(1)


########################
#  运行方式
########################
# python3 stress_test.py


def send_request():
    try:
        results=[]
        QUERY = queries[random.randint(0, len(queries) - 1)]['instruction']
        if len(QUERY)<80:
            add_query=QUERY*100
            add_query=add_query[:80-len(QUERY)-len("<think> 嗯")]
            QUERY=add_query+QUERY+"<think> 嗯"
        data = {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": QUERY}
            ],
            "stream": True
        }
        headers = {"Content-Type": "application/json"}
        server_start_time = time.time()
        response = requests.post(URL, json=data, headers=headers, stream=True)

        response.raise_for_status()
        token_nums=0
        ttft=0
        generate_start_time=time.time()
        # if response.status_code==200:
        for line in response.iter_lines():
            first_time=None
            token_nums+=1

                # print(f"开始时间: {first_time:.4f}s")
                # print(f"首字符生成时间: {first_time-start_time:.4f}s")
            # 处理 SSE 格式
            #print(line)
            decoded_line = line.decode("utf-8").strip()
            if decoded_line.startswith("data:"):
                json_str = decoded_line.split("data:", 1)[-1].strip()
                if '[DONE]' in json_str or token_nums>512:
                    break
                #print(f"json_str: {json_str}")
                try:
                    chunk = json.loads(json_str)
                    if token_nums==1:
                        first_time=time.time()
                        ttft=first_time-server_start_time
                    
                    content = chunk['choices'][0]['delta']['content']
                    results.append(content)
                    # print(content)  # 流式输出
                    # print("--------------------------------")
                except json.JSONDecodeError as e:
                    print(f"JSON 解析失败: {e}")
        end_time=time.time()
        latency = end_time - server_start_time
        decoded_tokens_nums=0
        for result in results:
            #print(len(tokenizer.encode(result)))
            decoded_tokens_nums+=len(tokenizer.encode(result))
        tps=decoded_tokens_nums/(end_time-generate_start_time)
        e2e_latency=latency
        tpop=(end_time-generate_start_time)/decoded_tokens_nums
        return latency,True,decoded_tokens_nums,tps,e2e_latency,tpop,ttft
        # else:
        #     return 0,False,0,0,0,0




    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")






def stress_test():
    concurrency = 1
    with open("stress_test.txt", "w") as f:
        print(f"{MODEL} 压力测试开始...")
        while concurrency <= MAX_CONCURRENT:
            #concurrent_time=time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(send_request) for _ in range(concurrency)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            #concurrent_time=time.time()-concurrent_time
            latencies = [res[0] for res in results if res[0] is not None]
            tokens = [res[2] for res in results if res[2] is not None]
            success_count = sum(1 for res in results if res[1])
            tps = [res[3] for res in results if res[3] is not None]
            #e2e_latency = [res[4] for res in results if res[4] is not None]
            tpop = [res[5] for res in results if res[5] is not None]
            ttft = [res[6] for res in results if res[6] is not None]
            #error_messages = [res[2] for res in results if res[2] is not None]
            error_rate = 1 - (success_count / concurrency)
            if latencies and tokens:
                avg_latency = sum(latencies) / len(latencies)
                avg_tokens = sum(tokens) / len(tokens)  
                avg_speed = sum(tokens) / sum(latencies)
                avg_tps = sum(tps) / len(tps)
                # avg_e2e_latency = sum(e2e_latency) / len(e2e_latency)
                avg_tpop = sum(tpop) / len(tpop)
                avg_ttft = sum(ttft) / len(ttft)
            else:
                avg_latency = 0
                avg_tokens = 0
                avg_speed = 0
            
            print(f"并发请求数: {concurrency}, 平均生成tokens: {avg_tokens:.4f}, 平均e2e耗时: {avg_latency:.4f}s, 平均tps: {round(avg_tps*concurrency, 4)} tokens/s, 平均tpop: {round(avg_tpop, 4)}s, 平均ttft: {round(avg_ttft, 4)}s,rps: {round(1/avg_latency, 4)}")
            f.write(f"并发请求数: {concurrency}, 平均生成tokens: {avg_tokens:.4f}, 平均e2e耗时: {avg_latency:.4f}s, 平均tps: {round(avg_tps*concurrency, 4)} tokens/s, 平均tpop: {round(avg_tpop, 4)}s, 平均ttft: {round(avg_ttft, 4)}s,rps: {round(1/avg_latency, 4)}\n")
            time.sleep(2)
            # if error_messages:
            #     for error in error_messages:
            #         print(f"接口调用异常: error={error}")

            if avg_latency > TIMEOUT_THRESHOLD or error_rate > ERROR_THRESHOLD:
                print(f"达到性能瓶颈，测试结束, 平均耗时={avg_latency}, 错误率={error_rate}")
                break
            if concurrency==1:
                concurrency+=1
            else:
                concurrency += 2


if __name__ == "__main__":
    stress_test()