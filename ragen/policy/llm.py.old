from typing import Optional
from vllm import LLM, SamplingParams
from ragen.env.sokoban import SokobanEnv
from ragen.policy.base import BasePolicy
from ragen.policy.bfs import BFSPolicy
from ragen.evaluators.trajectory_evaluator import TrajectoryEvaluator
from typing import List

class LLMPolicy(BasePolicy):
    def __init__(self, model_path: str, tensor_parallel_size: int = 1, temperature: float = 0.0):
        """Initialize the LLM-based policy.
        
        Args:
            model_path: Path to the trained model
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
        """
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            dtype="auto"
        )
        self.tokenizer = self.model.get_tokenizer()
        self.sampling_params = SamplingParams(
            temperature=temperature,  # Use greedy decoding
            max_tokens=128,    # Assuming maximum sequence length of 10 actions
            stop=None,
            top_p=0.95,
        )
    
    def format_observation(self, observation: str) -> str:
        """Not implemented now, just return the observation"""
        return observation
    
    def select_action(self, observation: str, env: Optional[SokobanEnv] = None) -> int:
        """Select an action based on the current observation using the LLM.
        
        Args:
            observation: The current observation from the environment
            env: Optional environment instance (not used in this implementation)
            
        Returns:
            Selected action (1=up, 2=down, 3=left, 4=right)
        """
        prompt = self.format_observation(observation)
        prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
        prompt = prompt + "<|im_start|>assistant\n"
        
        # Generate action prediction
        outputs = self.model.generate([prompt], self.sampling_params)
        prediction = outputs[0].outputs[0].text.strip()
        return prediction

    def select_action_multienv(self, observations: List[str], envs: List[SokobanEnv] = None):
        """Select an action based on the current observations using the LLM.
        
        Args:
            observations: List of current observations from the environments
            envs: List of environment instances

        Returns:
            List of selected actions for each environment
        """
        prompts = [self.format_observation(obs) for obs in observations]
        prompts = [self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False) for prompt in prompts]
        prompts = [prompt + "<|im_start|>assistant\n" for prompt in prompts]

        outputs = self.model.generate(prompts, self.sampling_params)
        predictions = [output.outputs[0].text.strip() for output in outputs]
        return predictions




class SokobanEvaluator:
    def __init__(self, llm_policy: LLMPolicy):
        """Initialize evaluator with a shared LLMPolicy instance."""
        self.llm_policy = llm_policy
    
    def basic_usage(self):
        """Example 1: Basic Usage - Single Step"""
        print("\nExample 1: Basic Usage - Single Step")
        env = SokobanEnv()
        
        observation = env.reset()
        print("Initial observation:")
        print(observation)
        
        action = self.llm_policy.select_action(observation)
        print(f"Selected action: {action} (1=up, 2=down, 3=left, 4=right)")
        
        next_obs, reward, done, info = env.step(action)
        print(f"Reward: {reward}, Done: {done}")
        print("New observation:")
        print(next_obs)

    def complete_episode(self, seed: int = 42, max_steps: int = 50):
        """Example 2: Complete Episode"""
        print("\nExample 2: Complete Episode")
        env = SokobanEnv()
        env.seed(seed)
        
        observation = env.reset()
        total_reward = 0
        steps = 0
        
        print("Starting episode...")
        while steps < max_steps:
            action = self.llm_policy.select_action(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            print(f"Step {steps}: Action={action}, Reward={reward}")
            if done:
                break
        
        print(f"Episode finished: Steps={steps}, Total Reward={total_reward}")
        print(f"Success: {env.success()}")

    def batch_evaluation(self, seeds=None):
        """Example 3: Batch Evaluation"""
        print("\nExample 3: Batch Evaluation")
            
        env = SokobanEnv()
        evaluator = TrajectoryEvaluator(env, self.llm_policy, max_steps=50)
        
        trajectories = evaluator.batch_evaluate(seeds=seeds if seeds else range(100), mp=False)
        evaluator.print_metrics()
        
        if trajectories:
            print("\nDetailed results for first trajectory:")
            traj = trajectories[0]
            print(f"Success: {traj['success']}")
            print(f"Steps taken: {len(traj['observations'])}")
            print(f"Total reward: {sum(traj['rewards'])}")

    def compare_with_bfs(self, seed: int = 42):
        """Example 4: Comparing with BFSPolicy"""
        print("\nExample 4: Comparing with BFSPolicy")
        env = SokobanEnv()
        env.seed(seed)
        
        # Evaluate LLMPolicy
        llm_evaluator = TrajectoryEvaluator(env, self.llm_policy, max_steps=100)
        llm_trajectories = llm_evaluator.batch_evaluate(seed=[seed])
        
        # Evaluate BFSPolicy
        bfs_policy = BFSPolicy()
        bfs_evaluator = TrajectoryEvaluator(env, bfs_policy, max_steps=100)
        bfs_trajectories = bfs_evaluator.batch_evaluate(seed=[seed])
        
        print("LLMPolicy Results:")
        llm_evaluator.print_metrics()
        print("\nBFSPolicy Results:")
        bfs_evaluator.print_metrics()

def main():
    print("LLMPolicy Usage Examples")
    print("=" * 50)
    
    # Initialize LLMPolicy once
    llm_policy = LLMPolicy(
        model_path=".cache/qwen2_5_3B_action_agent/lora_single_device/epoch_0"
    )
    
    # Create evaluator with shared policy
    evaluator = SokobanEvaluator(llm_policy)
    
    # Run all examples
    # evaluator.basic_usage()
    # evaluator.complete_episode()
    evaluator.batch_evaluation()
    evaluator.compare_with_bfs()
        
    
    print("\nAll examples completed!")

if __name__ == "__main__":
    main()