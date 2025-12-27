# 基于 Ray 集群架构的 vLLM 多机多卡大模型部署

本文档记录了在 **Ray 集群架构** 下，使用 **vLLM** 实现 **多机多卡（Pipeline Parallel + Tensor Parallel）大模型推理部署** 的完整流程，适用于内网多服务器环境。

---

## 一、环境准备

### 1. Conda 与 Python 环境

在 **Header 节点** 和 **Worker 节点** 上均创建并激活 **相同 Python 版本**（示例：Python 3.10）的 Conda 环境：

```bash
conda create -n vllm-ray python=3.10 -y
conda activate vllm-ray
```

### 2. 安装 Ray（版本必须一致）

以 Ray `2.53.0` 为例，**所有节点均需安装**：

```bash
pip install "ray[default]==2.53.0"
```

---

## 二、网络与 IP 配置

### 1. 查看网卡与 IP 地址

在每台服务器执行：

```bash
ip addr
```

记录以下信息：

- 本机 IP 地址
- 使用的物理网卡名称（如 `eno1`、`eth0` 等）

示例：

- Header 节点 IP：`192.*.*.107`
- Worker 节点 IP：`192.*.*.103`
- 使用网卡：`eno1`

### 2. 指定 NCCL / GLOO 通信网卡（所有节点）

```bash
export GLOO_SOCKET_IFNAME=eno1
export NCCL_SOCKET_IFNAME=eno1
```

> ⚠️ **强烈建议显式指定网卡**，否则在多网卡或 Docker / 虚拟网络环境下容易通信失败。

---

## 三、防火墙配置（关键步骤）

> ⚠️ **Ray + vLLM 多机通信依赖大量动态端口**，
> **只开放个别端口（如 6379）通常是不够的！**

### 原则

- **仅对集群节点 IP 互相放行**
- **放行全端口范围**
- **确保内网环境安全**

### Header → Worker（在 192.*.*.103 上执行）

```bash
sudo ufw allow from 192.*.*.107
sudo ufw reload
sudo ufw status
```

### Worker → Header（在 192.*.*.107 上执行）

```bash
sudo ufw allow from 192.*.*.103
sudo ufw reload
sudo ufw status
```

> 请使用 `sudo ufw status` **确认节点间端口已全部放行**。

---

## 四、启动 Ray 集群

### 1. 启动 Header 节点

```bash
ray stop --force
ray start \
  --head \
  --node-ip-address=192.*.*.107 \
  --port=6379 \
  --dashboard-host=0.0.0.0
```

### 2. 启动 Worker 节点

先测试端口连通性：

```bash
nc -vz 192.168.*.* 6379
```

确认无误后启动 Worker：

```bash
ray stop --force
ray start \
  --address=192.*.*.107:6379 \
  --node-ip-address=192.*.*.103
```

### 3. 检查集群状态（任一节点）

```bash
ray status
```

当看到 **多个节点 / GPU 资源被正确识别**（例如：2 GPUs 来自 2 个节点），说明 Ray 集群已成功建立。

---

## 五、安装 vLLM 并准备模型

### 1. 安装 vLLM（所有节点）

```bash
pip install vllm
```

### 2. 公司服务器建议配置（所有节点）

```bash
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
```

### 3. 模型准备（非常重要）

- **Header 和 Worker 节点必须：**
  - 下载 **同一个模型**
  - 使用 **完全相同的模型路径**
- Worker 节点 **不需要手动启动 vLLM**，但必须具备独立加载模型的能力

示例模型（ModelScope）：

```text
/home/nbw/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B
```

再次确认 Ray 集群状态：

```bash
ray status
```

---

## 六、在 Header 节点启动 vLLM（多机多卡）

仅在 **Header 节点** 执行：

```bash
vllm serve /home/nbw/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 10322 \
  --distributed-executor-backend ray \
  --max-model-len 512 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.8 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 2 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --dtype half
```

### 参数说明

- `pipeline-parallel-size`
  - **等于服务器节点数**（几台机器就设置几）
- `tensor-parallel-size`
  - **单机 GPU 数量**
- `dtype=half`
  - 使用 FP16，降低显存占用
- `max-model-len`
  - 可根据显存情况调整上下文长度

> ⚠️ **不要混淆 Pipeline Parallel 和 Tensor Parallel！**

---

## 七、验证 vLLM 是否使用多节点资源

### 1. 查看 Ray 资源占用

```bash
ray status
```

确认多个节点 GPU 资源被 vLLM 占用。

### 2. 查看 GPU 使用情况（所有节点）

```bash
nvidia-smi
```

若 **Header 和 Worker 节点 GPU 均显示 Ray / vLLM 进程占用**，说明多机多卡生效。

---

## 八、API 调用测试

在 **Header 节点** 新开一个终端，调用 OpenAI 兼容接口：

```bash
curl -X POST http://127.0.0.1:10322/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/nbw/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B",
    "messages": [
      { "role": "user", "content": "请用中文写一首四行诗，主题是冬天" }
    ],
    "temperature": 0.7,
    "max_tokens": 128,
    "enable_thinking": false
  }'
```

### 成功标志

- API 正常返回内容
- 两台服务器 `nvidia-smi` 均显示 GPU 被占用

✅ **说明 vLLM 多机多卡部署成功！**

---

## 九、常见问题提示

- Ray 连接异常：
  - 优先检查 **防火墙是否真的全放行**
  - 确认 `GLOO_SOCKET_IFNAME` / `NCCL_SOCKET_IFNAME`
- vLLM 只使用单机：
  - 检查 `pipeline-parallel-size`
  - 确认 Worker 节点模型路径一致

---

## 十、参考技术栈

- Ray Distributed Runtime
- vLLM Inference Engine
- NCCL / GLOO
- ModelScope

---

> 本 README 适用于 **内网 Ray + vLLM 多机多卡推理部署场景**，可直接作为团队部署规范文档使用。

