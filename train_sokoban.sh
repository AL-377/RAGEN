export DATA_DIR=data/sokoban
export DIM_X=6
export DIM_Y=6
export NUM_BOXES=1
export MAX_STEPS=5
export SEARCH_DEPTH=30

# export CUDA_VISIBLE_DEVICES=0
# export BASE_MODEL=Qwen/Qwen2.5-0.5B
# export EXPERIMENT_NAME=test-qwen2.5-0.5b

export CUDA_VISIBLE_DEVICES=0 # For multi-gpus, please make sure MICRO_BATCH_SIZE >= gpu_count
export BASE_MODEL=sft/output/sokoban-output-lora-qwen-2.5-0.5b-base/global_step_255_merged
export EXPERIMENT_NAME=test-qwen2.5-0.5b-base-lora-sft-kl-coef-0.0001


export MICRO_BATCH_SIZE=2
export TRAIN_BATCH_SIZE=128 # 256
export PPO_BATCH_SIZE=32 # 128
export MAX_START_LENGTH=400 # the first round prompt max length
export MAX_RESPONSE_LENGTH=100
export MAX_OBS_LENGTH=120
export MAX_TURNS=5
export NUM_UPDATE_PER_ROLL=1 # roll out for a batch, then the model do N times of update. Currently not implemented.
export LOG_MODE="['wandb']" # or 'console'
export GCP=True # gradient checkpointing
export N_GPUS=1
export ROLLOUT_TP_SIZE=1



export MULTI_PROCESSING=ray # only choice for now
export ENV_NAME=sokoban
export VLLM_ATTENTION_BACKEND=XFORMERS
export MAX_PROMPT_LENGTH=$(($MAX_START_LENGTH + $MAX_RESPONSE_LENGTH * ($MAX_TURNS - 1) + $MAX_OBS_LENGTH * $MAX_TURNS))
echo "MAX_PROMPT_LENGTH: $MAX_PROMPT_LENGTH"

python -m verl.trainer.main_ppo \
multi_processing=$MULTI_PROCESSING \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=$TRAIN_BATCH_SIZE \
data.val_batch_size=1312 \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$MAX_RESPONSE_LENGTH \
data.max_start_length=$MAX_START_LENGTH \
data.max_obs_length=$MAX_OBS_LENGTH \
data.shuffle_train_dataloader=True \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.model.enable_gradient_checkpointing=$GCP \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_BATCH_SIZE \
actor_rollout_ref.actor.ppo_micro_batch_size=$MICRO_BATCH_SIZE \
actor_rollout_ref.rollout.log_prob_micro_batch_size=$MICRO_BATCH_SIZE \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
actor_rollout_ref.ref.log_prob_micro_batch_size=$MICRO_BATCH_SIZE \
critic.optim.lr=1e-5 \
critic.model.path=$BASE_MODEL \
critic.ppo_micro_batch_size=$MICRO_BATCH_SIZE \
algorithm.kl_ctrl.kl_coef=0.0001 \
trainer.logger=$LOG_MODE \
+trainer.val_before_train=False \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=1 \
trainer.save_freq=100 \
trainer.test_freq=100 \
trainer.project_name=Agent-R1 \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.total_epochs=15 \
env.name=$ENV_NAME \
env.dim_x=$DIM_X \
env.dim_y=$DIM_Y \
env.num_boxes=$NUM_BOXES \
env.max_steps=$MAX_STEPS \
env.search_depth=$SEARCH_DEPTH \
max_turns=$MAX_TURNS \
logging.log_images=True \
logging.log_image_dir=.log.debug/trajectory \
logging.log_image_step_size=1 \
logging.log_n_image_per_batch=8 \
2>&1 | tee verl_demo.log


