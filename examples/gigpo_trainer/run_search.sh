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

# ========== 可配置参数（均可通过环境变量覆盖，例如 TRAIN_DATA_SIZE=8 ./run_search.sh）==========
# 数据
train_data_size=${TRAIN_DATA_SIZE:-16}
val_data_size=${VAL_DATA_SIZE:-32}
TRAIN_DATA=${TRAIN_DATA:-"./data/searchR1_processed_direct/train.parquet"}
VAL_DATA=${VAL_DATA:-"./data/searchR1_processed_direct/test.parquet"}

# 序列长度（适配 max_steps 轮数）
max_prompt_length=${MAX_PROMPT_LENGTH:-8192}
max_response_length=${MAX_RESPONSE_LENGTH:-512}

# PPO / 显存相关（A100 0.6B 约 60GB 峰值时可沿用当前默认）
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
rollout_log_prob_micro_batch_per_gpu=${ROLLOUT_LOG_PROB_MICRO_BATCH_PER_GPU:-16}
ref_log_prob_micro_batch_per_gpu=${REF_LOG_PROB_MICRO_BATCH_PER_GPU:-16}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.5}
# chunked prefill 要求 max_num_batched_tokens >= max_model_len (prompt + response)
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-$((max_prompt_length + max_response_length))}

# GiGPO
mode=${GIGPO_MODE:-"mean_std_norm"}
enable_similarity=${GIGPO_ENABLE_SIMILARITY:-True}
similarity_thresh=${GIGPO_SIMILARITY_THRESH:-0.9}

# 环境（search）
env_max_steps=${ENV_MAX_STEPS:-20}
env_history_length=${ENV_HISTORY_LENGTH:-20}
group_size=${GROUP_SIZE:-2}
env_search_url=${ENV_SEARCH_URL:-"http://127.0.0.1:8000/retrieve"}
env_embed_url=${ENV_EMBED_URL:-"http://127.0.0.1:8000/embed"}
env_use_information_gain_reward=${ENV_USE_INFORMATION_GAIN_REWARD:-True}
env_redundancy_penalty_lambda=${ENV_REDUNDANCY_PENALTY_LAMBDA:-0.5}
# 最终结果正确 vs 步级信息增益 权重：terminal 调高、step 调低，使“答对”为主目标
env_terminal_reward_scale=${ENV_TERMINAL_REWARD_SCALE:-10}
env_step_reward_scale=${ENV_STEP_REWARD_SCALE:-0.2}

# 模型与训练
MODEL_PATH=${MODEL_PATH:-"./models/Qwen/Qwen2.5-3B-Instruct"}
actor_lr=${ACTOR_LR:-1e-6}
actor_lr_warmup_ratio=${ACTOR_LR_WARMUP_RATIO:-0.1}
invalid_action_penalty_coef=${INVALID_ACTION_PENALTY_COEF:-0.01}
algorithm_gamma=${ALGORITHM_GAMMA:-0.95}
param_offload=${PARAM_OFFLOAD:-False}
optimizer_offload=${OPTIMIZER_OFFLOAD:-False}

# Trainer
trainer_save_freq=${TRAINER_SAVE_FREQ:-50}
trainer_test_freq=${TRAINER_TEST_FREQ:-50}
trainer_total_epochs=${TRAINER_TOTAL_EPOCHS:-1}
trainer_val_before_train=${TRAINER_VAL_BEFORE_TRAIN:-False}
trainer_log_val_generations=${TRAINER_LOG_VAL_GENERATIONS:-10}
trainer_val_only=${TRAINER_VAL_ONLY:-False}
trainer_project_name=${TRAINER_PROJECT_NAME:-verl_agent_search}
trainer_experiment_name=${EXPERIMENT_NAME:-gigpo_sim0.9_qwen2.5_3b_instruct}
trainer_n_gpus_per_node=${TRAINER_N_GPUS_PER_NODE:-1}
trainer_nnodes=${TRAINER_NNODES:-1}

# 环境变量一览（覆盖示例）:
# 数据: TRAIN_DATA_SIZE, VAL_DATA_SIZE, TRAIN_DATA, VAL_DATA
# 长度: MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH
# PPO/显存: PPO_MINI_BATCH_SIZE, PPO_MICRO_BATCH_SIZE_PER_GPU, ROLLOUT_LOG_PROB_MICRO_BATCH_PER_GPU, REF_LOG_PROB_MICRO_BATCH_PER_GPU, GPU_MEMORY_UTILIZATION, MAX_NUM_BATCHED_TOKENS
# GiGPO: GIGPO_MODE, GIGPO_ENABLE_SIMILARITY, GIGPO_SIMILARITY_THRESH
# 环境: ENV_MAX_STEPS, ENV_HISTORY_LENGTH, GROUP_SIZE, ENV_SEARCH_URL, ENV_EMBED_URL, ENV_USE_INFORMATION_GAIN_REWARD, ENV_REDUNDANCY_PENALTY_LAMBDA, ENV_TERMINAL_REWARD_SCALE, ENV_STEP_REWARD_SCALE
# 模型/训练: MODEL_PATH, EXPERIMENT_NAME, ACTOR_LR, ACTOR_LR_WARMUP_RATIO, INVALID_ACTION_PENALTY_COEF, ALGORITHM_GAMMA, PARAM_OFFLOAD, OPTIMIZER_OFFLOAD
# Trainer: TRAINER_SAVE_FREQ, TRAINER_TEST_FREQ, TRAINER_TOTAL_EPOCHS, TRAINER_VAL_BEFORE_TRAIN, TRAINER_LOG_VAL_GENERATIONS, TRAINER_VAL_ONLY, TRAINER_PROJECT_NAME, TRAINER_N_GPUS_PER_NODE, TRAINER_NNODES

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$actor_lr_warmup_ratio \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$param_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$optimizer_offload \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$rollout_log_prob_micro_batch_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ref_log_prob_micro_batch_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=$param_offload \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=$invalid_action_penalty_coef \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=$algorithm_gamma \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    algorithm.gigpo.enable_similarity=$enable_similarity \
    algorithm.gigpo.similarity_thresh=$similarity_thresh \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=$env_max_steps \
    env.rollout.n=$group_size \
    env.history_length=$env_history_length \
    env.search.search_url="$env_search_url" \
    env.search.embed_url="$env_embed_url" \
    env.search.use_information_gain_reward=$env_use_information_gain_reward \
    env.search.redundancy_penalty_lambda=$env_redundancy_penalty_lambda \
    env.search.terminal_reward_scale=$env_terminal_reward_scale \
    env.search.step_reward_scale=$env_step_reward_scale \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$trainer_project_name \
    trainer.experiment_name=$trainer_experiment_name \
    trainer.n_gpus_per_node=$trainer_n_gpus_per_node \
    trainer.nnodes=$trainer_nnodes \
    trainer.save_freq=$trainer_save_freq \
    trainer.test_freq=$trainer_test_freq \
    trainer.total_epochs=$trainer_total_epochs \
    trainer.val_before_train=$trainer_val_before_train \
    trainer.log_val_generations=$trainer_log_val_generations \
    trainer.val_only=$trainer_val_only $@