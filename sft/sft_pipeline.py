#!/usr/bin/env python3
import os
import argparse
import yaml
import subprocess
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SFTPipeline:
    """
    The pipeline for SFT:
        1. Generate SFT data
        2. Finetune the model using LoRA
        3. Merge the base model with LoRA weights
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env_type = config['sft']['env_type']
        self.base_model = config['model']['base_model']
        self.output_dir = config['sft']['output_dir']
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Environment-specific configurations
        self.env_configs = {
            'sokoban': {
                'DIM_X': config['sft']['sokoban']['dim_x'],
                'DIM_Y': config['sft']['sokoban']['dim_y'],
                'NUM_BOXES': config['sft']['sokoban']['num_boxes'],
                'MAX_STEPS': config['sft']['sokoban']['max_steps'],
                'SEARCH_DEPTH': config['sft']['sokoban']['search_depth']
            },
            'frozenlake': {
                'SIZE': config['sft']['frozenlake']['size'],
                'P': config['sft']['frozenlake']['p'],
                'IS_SLIPPERY': config['sft']['frozenlake']['is_slippery']
            }
        }

    def generate_data(self) -> None:
        """Generate SFT data for the specified environment."""
        logger.info(f"Generating SFT data for {self.env_type}")
        
        # Set environment variables based on env type
        if self.env_type in self.env_configs:
            for key, value in self.env_configs[self.env_type].items():
                os.environ[key] = str(value)
        
        data_gen_config = self.config['sft']['data_generation']
        output_dir = os.path.join('sft/data', self.env_type)
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            "python -m",
            f"sft.utils.generate_sft_verl_{self.env_type}",
            f"--env {self.env_type}",
            f"--algo {data_gen_config['algo']}",
            f"--seed {data_gen_config['seed']}",
            f"--output {output_dir}",
            f"--train_size {data_gen_config['train_size']}",
            f"--test_size {data_gen_config['test_size']}",
            f"--bfs_max_depths {data_gen_config['bfs_max_depths']}",
            f"--prefix {data_gen_config['prefix']}",
            f"--num_processes {data_gen_config['num_processes']}"
        ]
        
        subprocess.run(" ".join(cmd), shell=True, check=True)
        logger.info("Data generation completed")

    def finetune_model(self) -> str:
        """Finetune the model using LoRA."""
        logger.info("Starting model finetuning")
        
        training_config = self.config['sft']['training']
        lora_output_dir = os.path.join(self.output_dir, "lora_weights")
        os.makedirs(lora_output_dir, exist_ok=True)
        
        cmd = [
            "torchrun",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={training_config['num_gpus']}",
            "-m verl.trainer.fsdp_sft_trainer",
            f"data.train_files=sft/data/{self.env_type}/train.parquet",
            f"data.val_files=sft/data/{self.env_type}/test.parquet",
            "data.prompt_key=prompt",
            "data.response_key=response",
            f"data.max_length={training_config['max_length']}",
            f"optim.lr={training_config['learning_rate']}",
            f"data.train_batch_size={training_config['train_batch_size']}",
            f"data.micro_batch_size={training_config['micro_batch_size']}",
            f"model.partial_pretrain={self.base_model}",
            f"trainer.default_local_dir={lora_output_dir}",
            f"trainer.experiment_name={training_config['experiment_name']}",
            f"trainer.logger={training_config['logger']}",
            f"trainer.total_epochs={training_config['epochs']}",
            f"trainer.default_hdfs_dir=null", # NOTE hard code here
            f"trainer.validate_before_training={str(training_config.get('validate_before_training', True)).lower()}",
            f"model.lora_rank={training_config['lora_rank']}",
            f"model.lora_alpha={training_config['lora_alpha']}",
            f"model.target_modules={training_config['target_modules']}",
            f"model.enable_gradient_checkpointing={str(training_config.get('enable_gradient_checkpointing', False)).lower()}"
        ]

        log_file = os.path.join(lora_output_dir, "train.log")
        cmd.extend([f"2>&1 | tee {log_file}"])
        
        subprocess.run(" ".join(cmd), shell=True, check=True)
        logger.info("Model finetuning completed")
        return lora_output_dir

    def merge_model(self, lora_path: str) -> None:
        """Merge the base model with LoRA weights."""
        logger.info("Merging base model with LoRA weights")
        
        merged_model_path = os.path.join(self.output_dir, "merged_model")

        # read information from log file to find checkpoint with lowest validation loss
        log_file = os.path.join(lora_path, "train.log")
        min_val_loss = float('inf')
        best_step = 0
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "val/loss:" in line:
                    step = int(line.split("step:")[1].split(" -")[0])
                    val_loss = float(line.split("val/loss:")[1].strip())
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        best_step = step
        
        if best_step == 0:
            raise ValueError("No validation loss found in log file, finetuning failed")
        checkpoint_path = os.path.join(lora_path, f"global_step_{best_step}")
        print(f"Merging model from {checkpoint_path}")
        
        cmd = [
            "python -m sft.utils.merge_lora",
            f"--base_model_name={self.base_model}",
            f"--lora_model_path={checkpoint_path}",
            f"--output_path={merged_model_path}"
        ]
        subprocess.run(" ".join(cmd), shell=True, check=True)
        
        logger.info(f"Model merged and saved to {merged_model_path}")

    def run(self) -> None:
        """Run the complete SFT pipeline."""
        try:
            # Step 1: Generate SFT data
            self.generate_data()
            
            # Step 2: Finetune the model
            lora_path = self.finetune_model()
            
            # Step 3: Merge the model
            self.merge_model(lora_path)
            
            logger.info("SFT pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Error in SFT pipeline: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Run SFT Pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--env_type', type=str, required=None, default=None, help='Environment type')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if args.env_type is not None:
        config['sft']['env_type'] = args.env_type
    
    # Run pipeline
    pipeline = SFTPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()