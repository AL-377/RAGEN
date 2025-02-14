import numpy as np
from typing import List, Dict, Any, Optional
from .trajectory_transformations import DecisionMaking
from .env_utils import set_seed


def generate_trajectory(env, policy, max_steps: int = 50) -> List[Dict[str, Any]]:
    # using generate_trajectory_multienv with one environment
    return generate_trajectory_multienv(env, policy, None, max_steps)[0]
    

def _apply_conv_template(pair: tuple) -> List[Dict[str, str]]:
    """Format prompt/prediction pairs into conversation template.
    
    Args:
        pair: Tuple of (prompt, prediction)
        
    Returns:
        List of conversation message dictionaries
    """
    prompt, prediction = pair
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": prediction}
    ]

def _init_environments(env, seeds=None):
    """Initialize multiple environment instances with optional seeds."""
    env_instances = []
    trajectories = []
    
    if seeds:
        trajectories = [[] for _ in seeds]
        for seed in seeds:
            new_env = env.copy()
            new_env.reset(seed=seed)
            env_instances.append(new_env)
    else:
        trajectories = [[]]
        env_instances.append(env.copy())
        
    return env_instances, trajectories

def _create_initial_info(env_instances):
    """Create initial observation structures for each environment."""
    return [[{
        "all-observation-list": [env.render(mode='list')],
        "all-observation": env.render(),
        "all-action": [],
        "action": "-1"
    }] for env in env_instances]

def _update_env(env, step, info, action, policy_input, action_full, trajectory):
    """Process single step and update trajectory."""
    observation_list = env.render(mode='list')
    observation = env.render()
    _, reward, done, _ = env.step(action)
    success = env.success()
    next_observation = env.render()
    
    trajectory.append({
        'step': step,
        'observation': observation,
        'observation-list': observation_list,
        'action': action,
        'reward': reward,
        'next_observation': next_observation,
        'done': done,
        'success': success,
        'policy_input': policy_input,
        'action_full': action_full
    })
    info[0]['all-observation-list'].append(observation_list)
    info[0]['all-observation'] += "\n\n" + next_observation
    # hard code: only preserving at most 20 steps for all-information
    if len(info[0]['all-observation-list']) > 20:
        info[0]['all-observation'] = "\n\n".join(info[0]['all-observation'].split("\n\n")[-20:])


    info[0]['all-action'].append(action)
    info[0]['action'] = str(action)
    
    return env, info, trajectory, done

def _get_input_from_info(info, transformation):
    input_pairs = [transformation.generate_pairs(info)[-1] for info in info]
    input_pairs = [transformation.create_prompt(pair["condition"], pair["prediction"]) for pair in input_pairs]
    return [pair[0] for pair in input_pairs]

def generate_trajectory_multienv(
    env, 
    policy, 
    seeds: Optional[List[int]] = None, 
    max_steps: int = 50
) -> List[List[Dict[str, Any]]]:
    """
    Generate trajectories for multiple environments in parallel.
    env: Base environment instance
    policy: Policy instance to generate actions
    seeds: List of random seeds for environment initialization
    max_steps: Maximum steps per trajectory
    """
    env_instances, trajectories = _init_environments(env, seeds)
    transformation = DecisionMaking()
    history_info = _create_initial_info(env_instances)

    for step in range(max_steps):
        policy_inputs = _get_input_from_info(history_info, transformation)
        actions_full = policy.select_action_multienv(policy_inputs, env_instances)
        actions = env.postprocess_predictions(actions_full)

        # execute actions
        for idx, (env, action, info, policy_input, action_full, traj) in enumerate(zip(env_instances, actions, history_info, policy_inputs, actions_full, trajectories)):
            if type(action) != list: action = [action]
            if type(action_full) != list: action_full = [action_full]
            if type(policy_input) != list: policy_input = [policy_input]

            for a, a_full, pinput in zip(action, action_full, policy_input):
                if step >= max_steps: break
                env, info, traj, done = _update_env(env, step, info, a, pinput, a_full, traj) # env_instances, history_info, trajectories are updated each step
                
    return trajectories

