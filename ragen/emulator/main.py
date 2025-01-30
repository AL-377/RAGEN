import argparse
import gymnasium as gym
from tianshou.policy import RandomPolicy
from ragen.emulator.generator import TrajectoryConfig, TrajectoryGenerator

def parse_args():
    parser = argparse.ArgumentParser(description="Generate trajectories using specified environment and policy.")
    
    # Environment settings
    parser.add_argument("--env", type=str, default="CartPole-v1",
                      help="Gymnasium environment name (default: 'CartPole-v1')")
    parser.add_argument("--max_steps", type=int, default=1000,
                      help="Maximum steps per episode (default: 1000)")
    
    # Data generation settings
    parser.add_argument("--train_size", type=int, default=300,
                      help="Number of training trajectories to generate (default: 300)")
    parser.add_argument("--test_size", type=int, default=10,
                      help="Number of test trajectories to generate (default: 10)")
    parser.add_argument("--seed", type=int, default=10000,
                      help="Random seed (default: 10000)")
    
    # Parallel processing settings
    parser.add_argument("--num_envs", type=int, default=4,
                      help="Number of parallel environments (default: 4)")
    parser.add_argument("--no_mp", action="store_true",
                      help="Disable multiprocessing")
    
    # Template settings
    parser.add_argument("--template", type=str, default="qwen-instruct",
                      choices=["qwen-instruct", "base"],
                      help="Template type for formatting prompts (default: 'qwen-instruct')")
    
    # Output settings
    parser.add_argument("--output", type=str, default="data/trajectories",
                      help="Output directory for saving datasets (default: 'data/trajectories')")
    
    # Policy settings
    parser.add_argument("--policy", type=str, default="random",
                      choices=["random"],  # Can add more policies here
                      help="Policy type to use (default: 'random')")

    return parser.parse_args()

def create_policy(args, env):
    """Create policy based on arguments"""
    if args.policy == "random":
        return RandomPolicy(action_space=env.action_space)
    else:
        raise ValueError(f"Unsupported policy type: {args.policy}")

def main():
    args = parse_args()
    
    # Create configuration
    config = TrajectoryConfig(
        env_name=args.env,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
        num_envs=args.num_envs,
        template_type=args.template,
        max_steps=args.max_steps,
        use_multiprocessing=not args.no_mp
    )
    
    # Create environment and policy
    env = gym.make(args.env)
    policy = create_policy(args, env)
    
    # Create generator
    generator = TrajectoryGenerator(config, policy)
    
    # Generate datasets
    print(f"Generating trajectories for {args.env}...")
    print(f"Train size: {args.train_size}, Test size: {args.test_size}")
    
    train_dataset, test_dataset = generator.generate()
    # Save datasets
    print(f"Saving datasets to {args.output}...")
    generator.save_datasets(
        train_dataset,
        test_dataset,
        output_dir=args.output
    )
    
    print("Done!")
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")

if __name__ == "__main__":
    main()
