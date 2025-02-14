from abc import ABC, abstractmethod
import gymnasium as gym
import re
import numpy as np
from typing import Optional, List, Tuple, Any, Dict

class BaseEnv(ABC):
    """
    Abstract base class for all environments.
    The class needs to handle text-based input, input may be invalid
        - Environment will track the total reward for the trajectory

    """
    INVALID_ACTION = 0
    PENALTY_FOR_INVALID = -1
    def __init__(self):
        self.reward = 0

    
    
    @classmethod
    @abstractmethod
    def postprocess_predictions(cls, envs: List['BaseEnv'], predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            envs: List of environment instances
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        pass


    @abstractmethod
    def parse_update_info_to_obs(self, update_info: Tuple[Any, float, bool, Dict], action_is_valid: bool) -> str:
        """
        Parse environment update information into observation string.
        
        Args:
            update_info: Tuple of (observation, reward, done, info)
            action_is_valid: Whether the action was valid
            
        Returns:
            Formatted observation string
        """
        pass

    
    @classmethod
    def execute_predictions(cls, envs: List['BaseEnv'], predictions: List[str], pad_token: str) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, action_is_valid = cls.postprocess_predictions(envs, predictions)
        next_obs, dones = [], []
        
        for env, action, response, av in zip(envs, cur_actions, predictions, action_is_valid):
            obs = ""
            if "<|im_end|>" not in response:
                obs += "<|im_end|>"

            if env.success():
                obs += pad_token
                done = True
            else:
                observation, reward, done, extra_info = env.step(action)
                env_feedback = env.parse_update_info_to_obs(
                    (observation, reward, done, extra_info), 
                    av
                )
                env.reward += reward if av else (reward + env.PENALTY_FOR_INVALID)
                obs += "\n <|im_start|>user\n" + env_feedback + "<|im_end|>\n" + "<|im_start|>assistant\n<think>"
            next_obs.append(obs)
            dones.append(done)
        return next_obs, dones


    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Any:
        """
        Reset the environment.
        NOTE: the environment should be same for the same seed
        Args:
            seed: Seed for the environment
            
        Returns:
            rendered environment
        """
        pass

    @abstractmethod
    def step(self, action) -> Tuple[Any, float, bool, Dict]:
        """
        Execute one step in the environment.
        NOTE should also handle predefined invalid action (0)
        Args:
            action: Action to take, must be in action space, or default invalid action
            
        Returns:
            observation (rendered environment), reward, done, info
        """
        pass

    @abstractmethod
    def success(self) -> bool:
        """Check if the current state is successful."""
        pass

    @abstractmethod
    def render(self, mode: str = 'tiny_rgb_array') -> Any:
        """Render the environment."""
        pass

    @abstractmethod
    def copy(self) -> 'BaseEnv':
        """Create a deep copy of the environment."""
        pass








class BaseDiscreteActionEnv(BaseEnv, ABC):
    """
    Abstract base class for environments with discrete action spaces
    This class provides common functionality for environments like FrozenLakeEnv and SokobanEnv.
    """
    GRID_LOOKUP = {} # define the mapping from integer to string for rendering
    ACTION_LOOKUP = {} # define the mapping from integer to action string
    INVALID_ACTION = 0 # default invalid action
    PENALTY_FOR_INVALID = -1 # penalty for invalid action


    def parse_update_info_to_obs(self, update_info: Tuple[Any, float, bool, Dict], action_is_valid: bool) -> str:
        """
        Parse environment update information into observation string.
        
        Args:
            update_info: Tuple of (observation, reward, done, info)
            action_is_valid: Whether the action was valid
            
        Returns:
            Formatted observation string
        """
        observation, reward, done, _ = update_info
        if not action_is_valid:
            return f"Action is invalid. You stay in the same position. The observation is: \n{observation}\nreward: {reward}\ndone: {done}\n"
        return f"After you take this action, the observation is: \n{observation}\nreward: {reward}\ndone: {done}\n"


    @classmethod
    def postprocess_predictions(cls, envs: List['BaseDiscreteActionEnv'], predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            envs: List of environment instances
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        action_is_valid = []
        
        for env, prediction in zip(envs, predictions):
            if isinstance(prediction, str): # for llm output
                if "<answer>" in prediction:
                    action = prediction.split("<answer>")[1].split("</answer>")[0].strip()
                else:
                    action = prediction.strip()

                action = env.extract_action(action)
                action_is_valid.append(action != env.INVALID_ACTION)
            elif isinstance(prediction, int):
                action = prediction if prediction in env.get_all_actions() else env.INVALID_ACTION
                action_is_valid.append(action != env.INVALID_ACTION)
            elif isinstance(prediction, list):
                action = prediction
                action_is_valid.append(True)
            elif prediction is None:
                action = env.INVALID_ACTION
                action_is_valid.append(False)
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            
        return actions, action_is_valid

    def get_all_actions(self) -> List[int]:
        """Get list of all valid actions."""
        return list(range(self.ACTION_SPACE.start, self.ACTION_SPACE.start + self.ACTION_SPACE.n))
    





    @abstractmethod
    def extract_action(self, text: str) -> int:
        """
        Extract action (in action space) from text input.
        
        Args:
            text: Input text containing action
            
        Returns:
            Action in action space (otherwise a predefined invalid action)
        """
        pass

    @abstractmethod
    def reset(self, mode: str = 'tiny_rgb_array', seed: Optional[int] = None) -> Any:
        """
        Reset the environment.
        NOTE: the environment must be same for the same seed
        Args:
            mode: Mode to render the environment
            seed: Seed for the environment
            
        Returns:
            rendered environment
        """
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        """
        Execute one step in the environment.
        NOTE should also handle predefined invalid action (0)
        Args:
            action: Action to take, must be in action space, or default invalid action
            
        Returns:
            observation (rendered environment), reward, done, info
        """
        pass

    @abstractmethod
    def success(self) -> bool:
        """Check if the current state is successful."""
        pass

    @abstractmethod
    def render(self, mode: str = 'tiny_rgb_array') -> Any:
        """
        Render the environment.
        Args:
            mode: Mode to render the environment, needs to provide:
                - 'tiny_rgb_array': a string of the environment
                - 'rgb_array': a numpy array of the environment
        Returns:
            rendered environment, maybe a string or a numpy array (image)
        """
        pass

    @abstractmethod
    def copy(self) -> 'BaseDiscreteActionEnv':
        """Create a deep copy of the environment."""
        pass
