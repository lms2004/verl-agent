# SGLang Multi-turn 工具调用训练指南

本文档介绍如何配置和执行 GSM8K 多轮工具调用训练。

## 快速开始

### 1. 运行环境配置脚本

```bash
# 在项目根目录执行
bash docs/sglang_multiturn/multiturn/setup_gsm8k_multiturn.sh
```

该脚本会自动完成：
- 环境检查
- 模型下载
- 数据预处理
- 工具配置验证

### 2. 执行训练

```bash
bash tests/e2e/run_gsm8k_fsdp_sgl_multiturn_w_tool.sh
```

## 手动配置（可选）

如果自动配置脚本失败，可以手动执行以下步骤：

### 环境要求

- **GPU**: 8x H100 或同等算力
- **Python**: 3.10+
- **依赖**: verl 和 SGLang 已安装

### 安装依赖

```bash
pip install --upgrade pip
pip install -e ".[sglang]"
# 或
pip install -r requirements_sglang.txt
```

### 下载模型

```bash
PROJECT_DIR="$(pwd)"
huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
    --local-dir $PROJECT_DIR/models/Qwen/Qwen2.5-3B-Instruct
```

### 预处理数据

```bash
PROJECT_DIR="$(pwd)"
python3 examples/data_preprocess/gsm8k_multiturn_w_tool.py \
    --local_dir $PROJECT_DIR/data/gsm8k_verl_sgl_multi_turn_preprocessed
```

### 验证工具配置

```bash
ls -l examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml
```

## 执行训练

### 基本执行

```bash
# 确保在项目根目录
bash tests/e2e/run_gsm8k_fsdp_sgl_multiturn_w_tool.sh
```

### 自定义参数

```bash
bash tests/e2e/run_gsm8k_fsdp_sgl_multiturn_w_tool.sh \
    trainer.total_training_steps=100 \
    data.train_batch_size=128
```

### 后台执行

```bash
mkdir -p logs
nohup bash tests/e2e/run_gsm8k_fsdp_sgl_multiturn_w_tool.sh \
    > logs/gsm8k_multiturn_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## 常见问题

### 模型下载失败
- 检查网络连接
- 设置 HuggingFace token: `export HF_TOKEN=your_token`

### 数据预处理错误
- 确保 `datasets` 库已安装: `pip install datasets`
- 检查磁盘空间

### GPU 内存不足
- 减小 `data.train_batch_size`
- 减小 `actor_rollout_ref.rollout.n`
- 启用参数卸载: `actor_rollout_ref.actor.fsdp_config.param_offload=True`

## 相关文档

- [Multi-turn Rollout 配置](multiturn.rst)
- [Search Tool 集成示例](../search_tool_example.rst)
- [SGLang Worker 文档](../../workers/sglang_worker.rst)
