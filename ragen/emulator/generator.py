from typing import Dict, List, Optional, Tuple, Union, Callable
import gymnasium as gym
import numpy as np
from tianshou.data import Batch, Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import BasePolicy
import torch
from dataclasses import dataclass
from datasets import Dataset

@dataclass
class TrajectoryConfig:
    """Configuration for trajectory generation"""
    env_name: str
    train_size: int = 300
    test_size: int = 10
    seed: int = 10000
    num_envs: int = 4  # number of parallel environments
    template_type: str = 'qwen-instruct'
    max_steps: int = 1000
    use_multiprocessing: bool = True

class TemplateFormatter:
    """Handles different prompt templates"""
    
    TEMPLATES = {
        'qwen-instruct': (
            '<|im_start|>system\nYou are a helpful assistant. <|im_end|>\n'
            '<|im_start|>user\n{prompt}\nAlways output: <think> [Your thoughts] </think> '
            '<answer> [your answer] </answer> with no extra test. '
            'Strictly follow this format. <|im_end|>\n<|im_start|>assistant\n<think>'
        ),
        'base': (
            'A conversation between User and Assistant. The user asks a question, '
            'and the Assistant solves it.\nUser: {prompt}\nShow your work in '
            '<think> </think> tags.\nAssistant: \n<think>'
        )
    }

    @classmethod
    def format(cls, template_type: str, prompt: str) -> str:
        template = cls.TEMPLATES.get(template_type, cls.TEMPLATES['qwen-instruct'])
        return template.format(prompt=prompt)

class TrajectoryGenerator:
    """Generate trajectories using Tianshou framework"""
    
    def __init__(
        self,
        config: TrajectoryConfig,
        policy: BasePolicy,
        env_factory: Optional[Callable] = None
    ):
        self.config = config
        self.policy = policy
        
        # Set random seed
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Create environments
        if env_factory is None:
            env_factory = lambda: gym.make(config.env_name)
            
        if config.use_multiprocessing and config.num_envs > 1:
            self.envs = SubprocVectorEnv(
                [env_factory for _ in range(config.num_envs)]
            )
        else:
            self.envs = DummyVectorEnv(
                [env_factory for _ in range(config.num_envs)]
            )

        # 计算每个环境需要的平均轨迹数
        episodes_per_env = (config.train_size + config.test_size) // config.num_envs
        # 为每个环境分配足够大的buffer
        buffer_size_per_env = episodes_per_env * config.max_steps
        
        # Create buffer with enough capacity
        self.buffer = VectorReplayBuffer(
            total_size=buffer_size_per_env * config.num_envs,  # 总大小是每个环境的buffer大小乘以环境数量
            buffer_num=config.num_envs
        )
        
        # Initialize collector
        self.collector = Collector(
            policy=policy,
            env=self.envs,
            buffer=self.buffer,
            exploration_noise=True,
            preprocess_fn=self._preprocess_fn
        )

    def _preprocess_fn(self, obs=None, act=None, rew=None, done=None, info=None, policy=None, 
                      env_id=None, obs_next=None, terminated=None, truncated=None):
        """Preprocess function for collector."""
        if obs is None:
            return None
            
        # 创建一个标准的 Batch 对象
        result = Batch()
        
        if obs is not None:
            # 直接设置观察值，不使用嵌套结构
            result.obs = obs
            
        if act is not None:
            result.act = act
        if rew is not None:
            result.rew = rew
        if done is not None:
            result.done = done
        if info is not None:
            result.info = info
        if policy is not None:
            result.policy = policy
        if obs_next is not None:
            result.obs_next = obs_next
        if terminated is not None:
            result.terminated = terminated
        if truncated is not None:
            result.truncated = truncated
            
        return result

    def _format_observation(self, obs: np.ndarray) -> str:
        """Format observation into a string representation"""
        if isinstance(obs, np.ndarray):
            if len(obs.shape) > 1:
                return f"Image observation with shape {obs.shape}"
            return f"State: {obs.tolist()}"
        return str(obs)

    def _create_instance(
        self,
        idx: int,
        trajectory: Dict,
        split: str = "train"
    ) -> Dict:
        """Create a single dataset instance"""
        # Format observation
        obs = trajectory['obs']  # Changed from 'policy_input' to 'obs'
        obs_str = self._format_observation(obs)
        
        # Add action and reward information
        action = trajectory['act']  # Changed from 'action' to 'act'
        reward = trajectory['rew']  # Changed from 'reward' to 'rew'
        info = trajectory.get('info', {})
        
        # Create prompt with trajectory information
        prompt = (
            f"Environment: {self.config.env_name}\n"
            f"Observation: {obs_str}\n"
            f"Available actions: {self.envs.get_env_attr('action_space')[0]}\n"
            f"Task: Choose the best action based on the current state."
        )
        
        prompt_formatted = TemplateFormatter.format(
            self.config.template_type,
            prompt
        )

        return {
            "data_source": self.config.env_name,
            "prompt": [{"role": "user", "content": prompt_formatted}],
            "ability": self.policy.__class__.__name__.lower(),
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "target": float(reward),
                    "numbers": [float(reward), 0.0]
                }
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "trajectory": {
                    "observation": obs,
                    "action": int(action) if isinstance(action, (np.integer, np.floating)) else action,
                    "reward": float(reward),
                    "done": bool(trajectory['done']),
                    "info": dict(info)
                }
            }
        }

    def visualize_datasets(self, train_dataset: Dataset, test_dataset: Dataset) -> None:
        """Visualize the generated datasets.
        
        Args:
            train_dataset: The training dataset
            test_dataset: The testing dataset
        """
        print("\n=== Dataset Statistics ===")
        print(f"Training set size: {len(train_dataset)}")
        print(f"Testing set size: {len(test_dataset)}")
        
        # 分析轨迹统计信息
        train_rewards = [item['reward_model']['ground_truth']['target'] for item in train_dataset]
        test_rewards = [item['reward_model']['ground_truth']['target'] for item in test_dataset]
        
        train_actions = [item['extra_info']['trajectory']['action'] for item in train_dataset]
        test_actions = [item['extra_info']['trajectory']['action'] for item in test_dataset]
        
        print("\n=== Reward Statistics ===")
        print("Training set:")
        print(f"  Mean reward: {np.mean(train_rewards):.3f}")
        print(f"  Max reward: {np.max(train_rewards):.3f}")
        print(f"  Min reward: {np.min(train_rewards):.3f}")
        print(f"  Std reward: {np.std(train_rewards):.3f}")
        
        print("\nTesting set:")
        print(f"  Mean reward: {np.mean(test_rewards):.3f}")
        print(f"  Max reward: {np.max(test_rewards):.3f}")
        print(f"  Min reward: {np.min(test_rewards):.3f}")
        print(f"  Std reward: {np.std(test_rewards):.3f}")
        
        print("\n=== Action Distribution ===")
        unique_train_actions = np.unique(train_actions, return_counts=True)
        unique_test_actions = np.unique(test_actions, return_counts=True)
        
        print("Training set:")
        for action, count in zip(*unique_train_actions):
            print(f"  Action {action}: {count} times ({count/len(train_actions)*100:.1f}%)")
            
        print("\nTesting set:")
        for action, count in zip(*unique_test_actions):
            print(f"  Action {action}: {count} times ({count/len(test_actions)*100:.1f}%)")
        
        # 显示样本示例
        print("\n=== Sample Examples ===")
        print("\nTraining Example:")
        self._print_example(train_dataset[0])
        
        print("\nTesting Example:")
        self._print_example(test_dataset[0])

    def _print_example(self, example: Dict) -> None:
        """Print a single example in a readable format."""
        print("Prompt:")
        for message in example['prompt']:
            print(f"  {message['role']}: {message['content'][:200]}...")
        
        print("\nTrajectory Info:")
        traj = example['extra_info']['trajectory']
        print(f"  Observation: {traj['observation']}")
        print(f"  Action: {traj['action']}")
        print(f"  Reward: {traj['reward']}")
        print(f"  Done: {traj['done']}")
        
        print("\nReward Model:")
        print(f"  Target: {example['reward_model']['ground_truth']['target']}")
        print(f"  Style: {example['reward_model']['style']}")
        
        print("\nMetadata:")
        print(f"  Data Source: {example['data_source']}")
        print(f"  Split: {example['extra_info']['split']}")
        print(f"  Index: {example['extra_info']['index']}")

    def generate(self) -> Tuple[Dataset, Dataset]:
        """Generate trajectories and create train/test datasets"""
        
        # Reset everything
        self.collector.reset()
        self.collector.reset_env()
        self.collector.reset_buffer()
        
        # 初始化 collector 的数据
        obs, _ = self.envs.reset()
        self.collector.data = Batch(
            obs=obs,
            act=np.zeros(self.config.num_envs, dtype=np.int64),
            rew=np.zeros(self.config.num_envs, dtype=np.float32),
            done=np.zeros(self.config.num_envs, dtype=bool),
            obs_next=np.zeros_like(obs),
            info=[{} for _ in range(self.config.num_envs)],
            policy=Batch()
        )
        
        # 收集数据
        total_episodes = 0
        required_episodes = self.config.train_size + self.config.test_size
        
        print(f"Collecting {required_episodes} episodes...")
        
        while total_episodes < required_episodes:
            # 收集一个 episode
            result = self.collector.collect(n_episode=1, random=True)
            total_episodes += 1
            
            if total_episodes % 100 == 0:
                print(f"Collected {total_episodes}/{required_episodes} episodes")
                print(f"Buffer size: {len(self.buffer)}")

        print(f"Final buffer size: {len(self.buffer)}")
        print(f"Number of trajectories needed: {required_episodes}")
        
        # Get trajectories from buffer
        trajectories = []
        for i in range(len(self.buffer)):
            traj = {
                'obs': self.buffer.obs[i],
                'act': self.buffer.act[i],
                'rew': self.buffer.rew[i],
                'done': self.buffer.done[i],
                'info': self.buffer.info[i] if hasattr(self.buffer, 'info') else {}
            }
            trajectories.append(traj)
            
        print(f"Number of trajectories collected: {len(trajectories)}")

        if len(trajectories) < required_episodes:
            raise ValueError(
                f"Not enough trajectories collected. "
                f"Got {len(trajectories)}, need {required_episodes}"
            )

        # Create train and test datasets
        train_data = [
            self._create_instance(
                self.config.seed + i,
                trajectories[i],
                "train"
            )
            for i in range(self.config.train_size)
        ]
        
        test_data = [
            self._create_instance(
                self.config.seed + i,
                trajectories[i],
                "test"
            )
            for i in range(
                self.config.train_size,
                self.config.train_size + self.config.test_size
            )
        ]

        train_dataset = Dataset.from_list(train_data)
        test_dataset = Dataset.from_list(test_data)
        
        # 添加可视化
        self.visualize_datasets(train_dataset, test_dataset)

        return train_dataset, test_dataset

    def save_datasets(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        output_dir: str
    ) -> None:
        """Save datasets to parquet files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        train_dataset.to_parquet(
            os.path.join(output_dir, 'train.parquet')
        )
        test_dataset.to_parquet(
            os.path.join(output_dir, 'test.parquet')
        )
