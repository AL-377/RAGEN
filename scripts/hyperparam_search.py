from dataclasses import dataclass, field
from typing import Dict, List, Union, Any, Optional
import itertools
import subprocess
import os
from datetime import datetime
import argparse
from enum import Enum, auto
import re
import json
from pathlib import Path


# ============= Hyperparameter Configuration Part =============
class ParamType(Enum):
    NUMERIC = auto()
    CATEGORICAL = auto()
    BOOLEAN = auto()

@dataclass
class HyperParam:
    name: str
    param_type: ParamType
    default_value: Any
    search_space: Optional[List[Any]] = None
    group: int = 0
    
    def get_values(self, mode: str) -> List[Any]:
        """Get parameter values based on mode (search/fixed/default)"""
        if mode == "search" and self.search_space:
            return self.search_space
        return [self.default_value]

@dataclass
class HyperParamGroup:
    group_id: int
    params: List[HyperParam]
    description: str = ""

class HyperParamConfig:
    def __init__(self):
        self.groups: Dict[int, HyperParamGroup] = {}
        self.fixed_params: Dict[str, Any] = {
            "model.base_model": ["Qwen/Qwen2.5-0.5B-Instruct"],
            "optimization.adv_estimator": ["grpo"],
            "training.use_kl_loss": [True]
        }
        self._initialize_groups()

    def _initialize_groups(self):
        """Initialize all hyperparameter groups"""
        self.groups = {
            1: HyperParamGroup(1, [
                HyperParam("training.ppo_batch_size", ParamType.NUMERIC, 64, 
                          search_space=[16, 32, 64, 128, 256], group=1)
            ], "PPO Batch Size"), # 128
            
            2: HyperParamGroup(2, [
                HyperParam("training.train_batch_size", ParamType.NUMERIC, 8,
                          search_space=[8, 32, 64, 128, 256], group=2), # if bandits 128, if sokoban/fr 8
                HyperParam("training.n_rollout", ParamType.NUMERIC, 16,
                          search_space=[1, 2, 4, 8, 16], group=2) # if bandits 1, if sokoban/fr 16
            ], "Training Batch Size and Rollout"),
            
            3: HyperParamGroup(3, [
                HyperParam("training.kl_coef", ParamType.NUMERIC, 0.04,
                          search_space=[0.001, 0.005, 0.01, 0.04, 0.1, 0.5], group=3) # 0.04
            ], "KL Coefficient"),
            
            4: HyperParamGroup(4, [
                HyperParam("training.max_turns", ParamType.NUMERIC, 5,
                          search_space=[2, 5, 8], group=4), # if fr/sokoban 5, if bandits 1; fr/sokobaninference 10, 
                HyperParam("training.temperature", ParamType.NUMERIC, 1.0,
                          search_space=[0.1, 0.5, 1], group=4) # eval 0.7
            ], "Max Turns and Temperature"),
            
            5: HyperParamGroup(5, [
                HyperParam("training.actor_lr", ParamType.NUMERIC, 1e-6,
                          search_space=[1e-6, 5e-6, 1e-5, 5e-5], group=5) # 1e-6
            ], "Actor Learning Rate")
        }


# ============= Hyperparameter Search Part =============
class HyperParamSearch:
    def __init__(self, config: HyperParamConfig):
        self.config = config
        self.param_grid: Dict[str, List[Any]] = {}
        self.searching_params: List[str] = []
        self.log_name = ""
        self.search_group_id = -1
        self.searched_param_log_path = './log/searched_hyper_params'

    def setup_search_group(self, search_group: int, fixed_values: Dict[str, Any]) -> None:
        """Set up parameter grid based on search group and fixed values"""
        # Add fixed parameters
        self.param_grid.update(self.config.fixed_params)
        self.search_group_id = search_group
        
        if 'system.n_gpus' in fixed_values:
            self.param_grid['system.n_gpus'] = [fixed_values['system.n_gpus']]
        if 'training.micro_batch_size' in fixed_values:
            self.param_grid['training.micro_batch_size'] = [fixed_values['training.micro_batch_size']]
        # Process each group based on its relation to the search group
        for group_id, group in self.config.groups.items():
            if group_id < search_group:
                # Fixed values for previous groups
                for param in group.params:
                    if param.name in fixed_values:
                        self.param_grid[param.name] = [fixed_values[param.name]]
                    else:
                        raise ValueError(f"Missing fixed value for {param.name} in group {group_id}")
            
            elif group_id == search_group:
                # Search space for current group
                for param in group.params:
                    if param.search_space:
                        self.param_grid[param.name] = param.search_space
                        self.searching_params.append(param.name)
            
            else:
                # Default values for future groups
                for param in group.params:
                    self.param_grid[param.name] = [param.default_value]

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameter combinations"""
        if all(key in params for key in ["training.ppo_batch_size", "training.train_batch_size", "training.n_rollout"]):
            if params["training.ppo_batch_size"] > params["training.train_batch_size"] * params["training.n_rollout"]:
                print(f"Invalid combination: ppo_batch_size ({params['training.ppo_batch_size']}) > "
                      f"train_batch_size ({params['training.train_batch_size']}) * "
                      f"n_rollout ({params['training.n_rollout']})")
                return False
        return True

    def generate_command(self, params: Dict[str, Any], base_name: str, env_name: str = "sokoban") -> str:
        """Generate command string for running experiment"""
        param_parts = []
        for param_name in self.searching_params:
            if param_name in params:
                value = params[param_name]
                param_name_short = param_name.split('.')[-1]
                formatted_value = self._format_value(value)
                param_parts.append(f"{param_name_short}_{formatted_value}")

        timestamp = datetime.now().strftime("%Hh%Mm%Ssec")
        self.log_name = f"[hs]_{'_'.join(param_parts)}"
        experiment_name = f"{base_name}_{'_'.join(param_parts)}_{timestamp}"

        cmd_parts = [
            f"bash train.sh {env_name}",
            f"model.experiment_name={experiment_name}"
        ]
        
        for key, value in params.items():
            cmd_parts.append(f"{key}={value}")

        return " \\\n".join(cmd_parts)

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format parameter value for command string"""
        if isinstance(value, float):
            if value < 0.0001:
                return f"{value:.2e}".replace('-0', '-').replace('+0', '')
            return f"{value:.4f}".rstrip('0').rstrip('.')
        return str(value)

    def run_grid_search(self, base_experiment_name: str, dry_run: bool = True, env_name: str = "sokoban") -> None:
        """Run grid search with all parameter combinations"""
        combinations = [dict(zip(self.param_grid.keys(), vals)) 
                       for vals in itertools.product(*self.param_grid.values())]
        
        print(f"Total combinations: {len(combinations)}")
        
        log_dir = self._setup_log_directory()

        best_param_combination = None
        best_score_so_far = float('-inf')
        
        for i, params in enumerate(combinations, 1):
            if not self.validate_params(params):
                print(f"Skipping invalid combination: {params}")
                print('-' * 80)
                continue
                
            command = self.generate_command(params, base_experiment_name, env_name)
            print(f"Running combination {i}/{len(combinations)}:")
            print(command)
            print('-' * 80)
            
            if not dry_run:
                my_dict = self._run_experiment(command, log_dir)
                if my_dict is not None and my_dict['global_score/mean'] > best_score_so_far:
                    best_param_combination = params
                    best_score_so_far = my_dict['global_score/mean']
                    print(f"New best combination: {params} with score {best_score_so_far}")
        if not dry_run:
            if best_param_combination:
                print(f"Best combination found: {best_param_combination} with score {best_score_so_far}")
                assert self.search_group_id != -1, "search_group_id should not be -1"
                file_name = os.path.join(self.searched_param_log_path, f"searched_params_group_{self.search_group_id}.json")
                os.makedirs(self.searched_param_log_path, exist_ok=True)
                with open(file_name, 'w') as f:
                    json.dump(best_param_combination, f, indent=2)


    def _setup_log_directory(self) -> str:
        """Set up logging directory"""
        log_dir = "./log/terminal/hyper_param_search_logs/"
        
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def _run_experiment(self, command: str, log_dir: str) -> Union[Dict, None]:
        """Run single experiment with logging"""
        try:
            log_file = os.path.join(log_dir, 
                                  f"{self.log_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            with open(log_file, 'w') as f:
                subprocess.run(command, shell=True, check=True, stdout=f, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}, return code was {e.returncode}")
            if hasattr(e, 'output'):
                print(e.output)
            return None

        ## read the log file and return the result based on certain pattern
        ### an example is like "global_metrics {'global_score/mean': -3.55859375, 'global_score/max': 10.9, 'global_score/min': -5.5, 'global_score/std': 3.8139244745355065}"
        with open(log_file, 'r') as f:
            log_content = f.read()
        groups = re.findall(r"global_metrics\s+({[^}]+})", log_content)
        if groups:
            metric_values_dict = eval(groups[-1])
            print(metric_values_dict)
            return metric_values_dict
        return None


# ============= Main Functionality Part =============
def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(description='Hyperparameter search with group-based control')
    
    # Required arguments
    parser.add_argument('--env_name', type=str, required=True,
                      help='Environment name (e.g., sokoban)')
    parser.add_argument('--base_experiment_name', type=str, required=True,
                      help='Base name for the experiment')
    parser.add_argument('--search_group', type=int, required=True, choices=range(1, 6),
                      help='Group number to search (1-5)')
    
    # Optional arguments
    parser.add_argument('--dry_run', action='store_true',
                      help='Print commands without executing')
    
    # Group-specific parameters
    parser.add_argument('--ppo_batch_size', type=int,
                      help='Fixed value for PPO batch size (required for groups 2-5)')
    parser.add_argument('--train_batch_size', type=int,
                      help='Fixed value for training batch size (required for groups 3-5)')
    parser.add_argument('--n_rollout', type=int,
                      help='Fixed value for number of rollouts (required for groups 3-5)')
    parser.add_argument('--kl_coef', type=float,
                      help='Fixed value for KL coefficient (required for groups 4-5)')
    parser.add_argument('--max_turns', type=int,
                      help='Fixed value for max turns (required for group 5)')
    parser.add_argument('--temperature', type=float,
                      help='Fixed value for temperature (required for group 5)')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='Number of GPUs to use for training')
    parser.add_argument('--micro_batch_size', type=int, default=1,
                        help='Micro batch size for RAGEN training, must be greater than n_gpus')
    
    return parser

def read_searched_params(group_id: int, params: List[str]) -> Union[Dict[str, Any], None]:
    """Read searched parameters from JSON file"""
    file_name = os.path.join('./log/searched_hyper_params', f"searched_params_group_{group_id}.json")
    if not os.path.exists(file_name):
        assert False, f"searched params for group {group_id} not found, please run the search first"
    with open(file_name, 'r') as f:
        searched_params = json.load(f)
    return {param: searched_params[param] for param in params}

def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command line arguments based on search group requirements.
    Reads parameters from previous search groups if available.
    
    Args:
        args: Command line arguments namespace
    Raises:
        ValueError: If required arguments for the specified search group are missing
    """
    # Group 2 requirements
    if args.search_group > 1:
        params = read_searched_params(1, ["training.ppo_batch_size"])
        if params and args.ppo_batch_size is None:
            args.ppo_batch_size = params.get('training.ppo_batch_size')
            print("Read ppo_batch_size from searched params:", args.ppo_batch_size)
        if args.ppo_batch_size is None:
            raise ValueError("--ppo_batch_size required for groups 2-5")

    # Group 3 requirements
    if args.search_group > 2:
        params = read_searched_params(2, [
            "training.train_batch_size",
            "training.n_rollout"
        ])
        if params:
            if args.train_batch_size is None:
                args.train_batch_size = params.get('training.train_batch_size')
                print("Read train_batch_size from searched params:", args.train_batch_size)
            if args.n_rollout is None:
                args.n_rollout = params.get('training.n_rollout')
                print("Read n_rollout from searched params:", args.n_rollout)
        if args.train_batch_size is None or args.n_rollout is None:
            raise ValueError("--train_batch_size and --n_rollout required for groups 3-5")

    # Group 4 requirements
    if args.search_group > 3:
        params = read_searched_params(3, ["training.kl_coef"])
        if params and args.kl_coef is None:
            args.kl_coef = params.get('training.kl_coef')
            print("Read kl_coef from searched params:", args.kl_coef)
        if args.kl_coef is None:
            raise ValueError("--kl_coef required for groups 4-5")

    # Group 5 requirements
    if args.search_group > 4:
        params = read_searched_params(4, [
            "training.max_turns",
            "training.temperature"
        ])
        if params:
            if args.max_turns is None:
                args.max_turns = params.get('training.max_turns')
                print("Read max_turns from searched params:", args.max_turns)
            if args.temperature is None:
                args.temperature = params.get('training.temperature')
                print("Read temperature from searched params:", args.temperature)
        if args.max_turns is None or args.temperature is None:
            raise ValueError("--max_turns and --temperature required for group 5")

def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    validate_args(args)
    print("=" * 80)
    
    # Create config and search objects
    config = HyperParamConfig()
    search = HyperParamSearch(config)
    
    # Collect fixed values from arguments
    fixed_values = {}
    if args.ppo_batch_size is not None:
        fixed_values["training.ppo_batch_size"] = args.ppo_batch_size
    if args.train_batch_size is not None:
        fixed_values["training.train_batch_size"] = args.train_batch_size
    if args.n_rollout is not None:
        fixed_values["training.n_rollout"] = args.n_rollout
    if args.kl_coef is not None:
        fixed_values["training.kl_coef"] = args.kl_coef
    if args.max_turns is not None:
        fixed_values["training.max_turns"] = args.max_turns
    if args.temperature is not None:
        fixed_values["training.temperature"] = args.temperature
    if args.n_gpus is not None:
        fixed_values["system.n_gpus"] = args.n_gpus
    if args.micro_batch_size is not None:
        fixed_values["training.micro_batch_size"] = args.micro_batch_size
        assert args.micro_batch_size >= args.n_gpus, "micro_batch_size must be greater than n_gpus"
    elif args.micro_batch_size is None and args.n_gpus is not None:
        assert 1 >= args.n_gpus, "micro_batch_size must be greater than n_gpus"    
    # Set up search configuration
    search.setup_search_group(args.search_group, fixed_values)
    
    # Run the search
    search.run_grid_search(
        base_experiment_name=args.base_experiment_name,
        dry_run=args.dry_run,
        env_name=args.env_name
    )

if __name__ == "__main__":
    main()