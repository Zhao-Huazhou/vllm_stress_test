## 部署模型

### 下载模型

```
git lfs clone https://huggingface.co/cognitivecomputations/DeepSeek-R1-AWQ
```

### 安装vllm

```
pip install vllm
```

### 启动 API 服务

``` bash
python3 -m vllm.entrypoints.openai.api_server \
--served-model-name deepseek-r1 \
--model ./DeepSeek-R1-AWQ \
--trust-remote-code \
--host 0.0.0.0 \
--port 8080 \
--max-model-len 2048 \
--tensor-parallel-size 8 \
--gpu_memory_utilization 0.9 \
--dtype bfloat16
```




## 压力测试

### 安装依赖

```
pip install requests transformers
```


### 运行脚本

```
python3 stress.py
```



