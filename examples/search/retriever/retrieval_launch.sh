save_path=./data/searchR1

index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

# Auto-detect available GPUs
# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    # Fallback: try to detect via Python
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
fi

# Function to find GPU with lowest memory usage
find_least_used_gpu() {
    if command -v nvidia-smi &> /dev/null && [ "$NUM_GPUS" -gt 1 ]; then
        # Get memory usage for each GPU and find the one with lowest usage
        LEAST_USED_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
            awk '{print $1, $2}' | \
            sort -k2 -n | \
            head -1 | \
            awk '{print $1}')
        echo "$LEAST_USED_GPU"
    else
        # Default to GPU 1 if multiple GPUs available, otherwise GPU 0
        # This allows retrieval server to use GPU 1 while training uses GPU 0
        if [ "$NUM_GPUS" -gt 1 ]; then
            echo "1"
        else
            echo "0"
        fi
    fi
}

# Configure GPU settings
# If user sets FAISS_GPU_ID or RETRIEVER_DEVICE via environment variable, use those
# Otherwise, auto-configure based on available GPUs
USE_FAISS_GPU=false
if [ "$NUM_GPUS" -gt 0 ]; then
    USE_FAISS_GPU=true
    if [ -z "$FAISS_GPU_ID" ]; then
        # Auto-select GPU: prefer GPU 1 if available (to avoid conflict with training on GPU 0)
        # Otherwise use the least used GPU
        if [ "$NUM_GPUS" -gt 1 ]; then
            # Prefer GPU 1 for retrieval server to avoid conflict with training
            FAISS_GPU_ID=1
            echo "Auto-detected $NUM_GPUS GPU(s), using GPU 1 for FAISS (to avoid conflict with training on GPU 0)"
        else
            FAISS_GPU_ID=0
            echo "Auto-detected $NUM_GPUS GPU(s), using GPU 0 for FAISS"
        fi
    else
        # Validate user-specified GPU ID
        if [ "$FAISS_GPU_ID" -ge "$NUM_GPUS" ]; then
            echo "Error: Specified GPU $FAISS_GPU_ID is not available. Only $NUM_GPUS GPU(s) available (0-$((NUM_GPUS-1)))"
            exit 1
        fi
        echo "Using user-specified GPU $FAISS_GPU_ID for FAISS"
    fi
    
    if [ -z "$RETRIEVER_DEVICE" ]; then
        RETRIEVER_DEVICE="cuda:$FAISS_GPU_ID"
        echo "Using GPU $FAISS_GPU_ID for retriever encoder"
    else
        echo "Using user-specified device: $RETRIEVER_DEVICE"
    fi
else
    echo "No GPUs detected, using CPU mode"
    if [ -z "$RETRIEVER_DEVICE" ]; then
        RETRIEVER_DEVICE="cpu"
    fi
    # Don't use FAISS GPU if no GPUs available
    USE_FAISS_GPU=false
fi

# Build command arguments
FAISS_GPU_ARGS=""
if [ "$USE_FAISS_GPU" = true ]; then
    FAISS_GPU_ARGS="--faiss_gpu --faiss_gpu_id $FAISS_GPU_ID"
fi

python examples/search/retriever/retrieval_server.py \
  --index_path $index_file \
  --corpus_path $corpus_file \
  --topk 3 \
  --retriever_name $retriever_name \
  --retriever_model $retriever_path \
  $FAISS_GPU_ARGS \
  --device $RETRIEVER_DEVICE \
  --port 8000 \