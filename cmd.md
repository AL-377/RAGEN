## Base Experiments

### Bandit

```bash
mkdir -p ./log/terminal

bash train.sh two_armed_bandit \
    model.base_model=Qwen/Qwen2.5-3B-Instruct \
    model.experiment_name=two_armed_bandit_ragen_main \
    training.micro_batch_size=4 \
    training.train_batch_size=128 \
    training.ppo_batch_size=128 \
    training.max_turns=1 \
    training.n_rollout=1 \
    training.total_training_steps=500 \
    training.use_kl_loss=False \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=brpo > ./log/terminal/two_armed_bandit_ragen_main.log
```

### Sokoban

```bash
mkdir -p ./log/terminal

bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-3B-Instruct \
    model.experiment_name=sokoban_ragen_main \
    training.micro_batch_size=4 \
    training.train_batch_size=8 \
    training.ppo_batch_size=128 \
    training.max_turns=5 \
    training.n_rollout=16 \
    training.total_training_steps=500 \
    training.use_kl_loss=False \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=grpo > ./log/terminal/sokoban_ragen_main.log
```

### FrozenLake

```bash
mkdir -p ./log/terminal
bash train.sh frozenlake \
    model.base_model=Qwen/Qwen2.5-3B-Instruct \
    model.experiment_name=frozenlake_ragen_main \
    training.micro_batch_size=4 \
    training.train_batch_size=8 \
    training.ppo_batch_size=128 \
    training.max_turns=5 \
    training.n_rollout=16 \
    training.total_training_steps=500 \
    training.use_kl_loss=False \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=grpo > ./log/terminal/frozenlake_ragen_main.log
```
