save_path=./data/searchR1

index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

# Specify GPU device for retrieval server (e.g., use GPU 1 if training uses GPU 0)
# Set to "cpu" to use CPU instead, or "cuda:1" to use GPU 1
RETRIEVER_DEVICE=${RETRIEVER_DEVICE:-"cuda:1"}
# Specify GPU ID for FAISS (e.g., 1). Leave empty to use all GPUs when --faiss_gpu is set
FAISS_GPU_ID=${FAISS_GPU_ID:-1}

python examples/search/retriever/retrieval_server.py \
  --index_path $index_file \
  --corpus_path $corpus_file \
  --topk 3 \
  --retriever_name $retriever_name \
  --retriever_model $retriever_path \
  --faiss_gpu \
  --faiss_gpu_id $FAISS_GPU_ID \
  --device $RETRIEVER_DEVICE \
  --port 8000 \