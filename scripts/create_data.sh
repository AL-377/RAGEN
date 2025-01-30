
python ragen/emulator/main.py \
    --env CartPole-v1 \
    --train_size 1000 \
    --test_size 100 \
    --num_envs 8 \
    --template qwen-instruct \
    --output data/cartpole_large