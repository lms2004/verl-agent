#!/bin/bash
# GSM8K Multi-turn 训练环境配置脚本
#
# 功能：自动完成环境检查、模型下载、数据预处理和工具配置验证
#
# 使用方法：
#   bash docs/sglang_multiturn/multiturn/setup_gsm8k_multiturn.sh
#
# 可配置参数（通过环境变量）：
#   - MODEL_NAME: 模型名称，默认 "Qwen/Qwen2.5-3B-Instruct"
#   - HF_ENDPOINT: HuggingFace 镜像地址，默认 "https://hf-mirror.com"
#   - DATA_DIR: 数据保存目录，默认 "$PROJECT_DIR/data/gsm8k_verl_sgl_multi_turn_preprocessed"
#
# 示例：
#   MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" bash setup_gsm8k_multiturn.sh

set -e

# 颜色输出函数
print_info() { echo -e "\033[0;32m[INFO]\033[0m $1"; }
print_warn() { echo -e "\033[1;33m[WARN]\033[0m $1"; }
print_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; }

# 获取项目根目录
[ -f "setup.py" ] || [ -f "pyproject.toml" ] || { print_error "请在项目根目录执行"; exit 1; }
PROJECT_DIR="$(pwd)"
print_info "项目目录: $PROJECT_DIR"

# ============================================================================
# 步骤 1: 环境检查与依赖安装
# ============================================================================
print_info "=== 步骤 1/4: 环境检查 ==="

# 检查 Python（实际命令: python3 --version）
python3 --version > /dev/null || { print_error "Python3 未安装"; exit 1; }
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_info "Python 版本: $PYTHON_VERSION"

# 检查并安装 verl（实际命令: pip install -e ".[sglang]" 或 pip install -r requirements_sglang.txt）
if ! python3 -c "import verl" 2>/dev/null; then
    print_warn "安装 verl..."
    # 实际执行: pip3 install --upgrade pip
    pip3 install --upgrade pip > /dev/null 2>&1 || true
    # 实际执行: pip3 install -r requirements_sglang.txt 或 pip3 install -e ".[sglang]"
    [ -f "requirements_sglang.txt" ] && pip3 install -r requirements_sglang.txt || pip3 install -e ".[sglang]"
fi
print_info "✓ verl 已安装"

# 检查并安装 SGLang（实际命令: pip install -e ".[sglang]"）
python3 -c "import sglang" 2>/dev/null || {
    print_warn "安装 SGLang..."
    # 实际执行: pip3 install -e ".[sglang]"
    pip3 install -e ".[sglang]" > /dev/null 2>&1 || pip3 install "sglang[all]==0.4.6.post5"
}
print_info "✓ SGLang 已安装"

# 检查并安装 datasets（实际命令: pip install datasets）
python3 -c "import datasets" 2>/dev/null || {
    print_warn "安装 datasets..."
    # 实际执行: pip3 install datasets
    pip3 install datasets
}
print_info "✓ datasets 已安装"

# 检查并安装 huggingface-cli（实际命令: pip install huggingface_hub）
command -v huggingface-cli > /dev/null || {
    print_warn "安装 huggingface-cli..."
    # 实际执行: pip3 install huggingface_hub
    pip3 install huggingface_hub
}
print_info "✓ huggingface-cli 已安装"

# ============================================================================
# 步骤 2: 下载模型
# ============================================================================
print_info "=== 步骤 2/4: 模型下载 ==="

# 配置参数（可通过环境变量覆盖）
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-3B-Instruct"}
MODEL_DIR="$PROJECT_DIR/models/$MODEL_NAME"

# 如果模型已存在则跳过下载
if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    print_info "模型已存在: $MODEL_DIR"
else
    print_info "下载模型: $MODEL_NAME"
    mkdir -p "$(dirname $MODEL_DIR)"
    
    # 设置 HuggingFace 镜像（可通过环境变量 HF_ENDPOINT 覆盖）
    export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}
    print_info "使用镜像: $HF_ENDPOINT"
    
    # 实际执行: huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir $MODEL_DIR
    huggingface-cli download $MODEL_NAME --local-dir $MODEL_DIR
    print_info "✓ 模型下载完成: $MODEL_DIR"
fi

# ============================================================================
# 步骤 3: 数据预处理
# ============================================================================
print_info "=== 步骤 3/4: 数据预处理 ==="

# 配置数据目录（可通过环境变量 DATA_DIR 覆盖）
DATA_DIR=${DATA_DIR:-"$PROJECT_DIR/data/gsm8k_verl_sgl_multi_turn_preprocessed"}
TRAIN_FILE="$DATA_DIR/train.parquet"
TEST_FILE="$DATA_DIR/test.parquet"

# 如果数据已存在则跳过预处理
if [ -f "$TRAIN_FILE" ] && [ -f "$TEST_FILE" ]; then
    print_info "数据已存在: $TRAIN_FILE, $TEST_FILE"
else
    print_info "预处理数据..."
    PREPROCESS_SCRIPT="$PROJECT_DIR/examples/data_preprocess/gsm8k_multiturn_w_tool.py"
    [ -f "$PREPROCESS_SCRIPT" ] || { print_error "预处理脚本不存在: $PREPROCESS_SCRIPT"; exit 1; }
    
    mkdir -p "$DATA_DIR"
    # 实际执行: python3 examples/data_preprocess/gsm8k_multiturn_w_tool.py --local_dir $DATA_DIR
    python3 "$PREPROCESS_SCRIPT" --local_dir "$DATA_DIR"
    print_info "✓ 数据预处理完成: $TRAIN_FILE, $TEST_FILE"
fi

# ============================================================================
# 步骤 4: 工具配置验证
# ============================================================================
print_info "=== 步骤 4/4: 工具配置验证 ==="

TOOL_CONFIG="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml"

# 检查工具配置文件是否存在
[ -f "$TOOL_CONFIG" ] || { print_error "工具配置不存在: $TOOL_CONFIG"; exit 1; }
print_info "✓ 工具配置存在: $TOOL_CONFIG"

# 验证 YAML 格式（如果 pyyaml 可用）
python3 -c "import yaml; yaml.safe_load(open('$TOOL_CONFIG'))" 2>/dev/null && \
    print_info "✓ 配置文件格式正确" || print_warn "无法验证 YAML 格式"

# ============================================================================
# 配置完成摘要
# ============================================================================
print_info "=== 环境配置完成 ==="
echo ""
echo "配置摘要:"
echo "  ✓ Python: $PYTHON_VERSION"
echo "  ✓ 模型: $MODEL_DIR"
echo "  ✓ 训练数据: $TRAIN_FILE"
echo "  ✓ 测试数据: $TEST_FILE"
echo "  ✓ 工具配置: $TOOL_CONFIG"
echo ""
print_info "开始训练:"
echo "  bash tests/e2e/run_gsm8k_fsdp_sgl_multiturn_w_tool.sh"
echo ""
print_info "自定义训练参数示例:"
echo "  bash tests/e2e/run_gsm8k_fsdp_sgl_multiturn_w_tool.sh \\"
echo "    trainer.total_training_steps=100 \\"
echo "    data.train_batch_size=128 \\"
echo "    actor_rollout_ref.rollout.gpu_memory_utilization=0.6"
echo ""
