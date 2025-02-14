import json
from collections.abc import MutableSequence
from .trajectory_transformations import TRANSFORMATION_REGISTRY
from .helpers import set_seed
from tqdm import tqdm
from random import sample
import os

class Dataset(MutableSequence):
    def __init__(
        self, data, transform=True, max_examples_per_traj=3, random_seed=42
    ):
        self.original_data = data
        if transform:
            self.transformed_data = self._transform_data(max_examples_per_traj, random_seed)
        else:
            self.transformed_data = data

    def _transform_data(
        self, max_examples_per_traj=3, random_seed=42
    ):            
        with set_seed(random_seed):
            transformed = [
                {
                    **instance,
                    "all-observation": "\n\n".join([inst["observation"] for inst in game[:i + 1]]),
                    "all-observation-list": [inst["observation-list"] for inst in game[:i + 1]],
                    "best_future_trajectory": [
                        (inst["observation"], inst["action"]) 
                        for inst in game[i:] 
                    ]
                }
                for game in self.original_data
                for i, instance in sample(list(enumerate(game)), min(len(game), max_examples_per_traj))
            ]
        return transformed
    
    def _apply_conv_template(self, prompt, prediction):
        return [{
            "role": "user",
            "content": prompt
        }, {
            "role": "assistant",
            "content": prediction
        }]
        
    def transform(self, strategy: str or list):
        """
        Transform the dataset using a specified strategy.
        """
        transformed = []
        
        if type(strategy) == str:
            strategy = [strategy] 
        if "all" in strategy:
            strategy = list(TRANSFORMATION_REGISTRY.keys())
                
        for s in strategy:
            assert s in TRANSFORMATION_REGISTRY, f"Strategy '{s}' is not defined."
            strategy_class = TRANSFORMATION_REGISTRY[s]()

            for example in tqdm(self.transformed_data, desc=f"Transforming with strategy '{s}'"):
                pairs = strategy_class.generate_pairs([example])
                for pair in pairs:
                    prompt, prediction = strategy_class.create_prompt(pair["condition"], pair["prediction"])
                    conversations = self._apply_conv_template(prompt, prediction)
                    transformed.append({
                        "strategy": s,
                        "item_id": example.get("item_id", ""),
                        "prompt": prompt,
                        "prediction": prediction,
                        "conversations": conversations,
                        "metadata": {
                            "prompt_length": len(prompt.split()),
                            "prediction_length": len(prediction.split()),
                            # "best_future_trajectory": example["best_future_trajectory"],
                        }
                    })

        return Dataset(transformed, transform=False)

    def __getitem__(self, key):
        return self.transformed_data[key] if isinstance(key, int) or isinstance(key, slice) else [i[key] for i in self.transformed_data]

    def __setitem__(self, index, value):
        self.transformed_data[index] = value

    def __delitem__(self, index):
        del self.transformed_data[index]

    def insert(self, index, value):
        self.transformed_data.insert(index, value)

    def __len__(self):
        return len(self.transformed_data)

    def save(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.transformed_data, f, indent=4)

    @classmethod
    def from_file(cls, file_path):
        with open(file_path, "r") as f:
            return cls(json.load(f))
