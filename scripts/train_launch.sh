export DATA_DIR=data/sokoban
export DIM_X=6
export DIM_Y=6
export NUM_BOXES=1
export MAX_STEPS=5
export SEARCH_DEPTH=30

# export CUDA_VISIBLE_DEVICES=0
# export BASE_MODEL=Qwen/Qwen2.5-0.5B
# export EXPERIMENT_NAME=test-qwen2.5-0.5b

export CUDA_VISIBLE_DEVICES=0
export BASE_MODEL=checkpoints/Agent-R1/test-qwen2.5-0.5b-instruct-1mbsz/actor/global_step_100
export EXPERIMENT_NAME=test-qwen2.5-0.5b-imagetest


export MICRO_BATCH_SIZE=1
export TRAIN_BATCH_SIZE=128 # 256
export PPO_BATCH_SIZE=64 # 128
export MAX_START_LENGTH=400 # the first round prompt max length
export MAX_RESPONSE_LENGTH=100
export MAX_OBS_LENGTH=120
export MAX_TURNS=5
export NUM_UPDATE_PER_ROLL=1 # roll out for a batch, then the model do N times of update. Currently not implemented.
export LOG_MODE="['wandb']" # or 'console'
export GCP=True # gradient checkpointing
export N_GPUS=1
export ROLLOUT_TP_SIZE=1

bash ./train.sh # more arguments in this file

# default config file is verl/trainer/config/ppo_trainer.yaml
