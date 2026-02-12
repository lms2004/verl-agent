set -x

ENGINE=${1:-vllm}

# Auto-detect and configure GPU settings for training
# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    # Fallback: try to detect via Python
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
fi

# Configure GPU selection for training
# If user sets CUDA_VISIBLE_DEVICES via environment variable, use that
# Otherwise, auto-configure: prefer GPU 0 (to avoid conflict with retrieval server on GPU 1)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    if [ "$NUM_GPUS" -gt 1 ]; then
        # Use GPU 0 for training (retrieval server will use GPU 1)
        export CUDA_VISIBLE_DEVICES=0
        echo "Auto-detected $NUM_GPUS GPU(s), using GPU 0 for training (retrieval server should use GPU 1)"
    elif [ "$NUM_GPUS" -eq 1 ]; then
        export CUDA_VISIBLE_DEVICES=0
        echo "Auto-detected 1 GPU, using GPU 0 for training"
    else
        echo "Warning: No GPUs detected, training will use CPU (may be slow)"
    fi
else
    echo "Using user-specified CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

train_data_size=32
val_data_size=64
group_size=2

mode="mean_std_norm"
enable_similarity=True
similarity_thresh=0.9

MODEL_PATH="./models/Qwen/Qwen2.5-3B-Instruct"

TRAIN_DATA="./data/searchR1_processed_direct/train.parquet"
VAL_DATA="./data/searchR1_processed_direct/test.parquet"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    algorithm.gigpo.enable_similarity=$enable_similarity \
    algorithm.gigpo.similarity_thresh=$similarity_thresh \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=$group_size \
    env.history_length=4 \
    env.search.search_url='http://127.0.0.1:8000/retrieve' \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_search' \
    trainer.experiment_name='gigpo_sim0.9_qwen2.5_3b_instruct' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False $@