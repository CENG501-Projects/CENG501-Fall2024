import tomllib as toml
import copy
from pprint import pprint

from src.experiments.experiment_runner import experiment_run

class Config:
    def __init__(self, input_dict) -> None:
        for top_key, sub_dict in input_dict.items():
            if isinstance(sub_dict, dict):
                for sub_key, value in sub_dict.items():
                    setattr(self, sub_key, value)


def load_config(config_path):
    with open(config_path, "rb") as f:
        config_toml = toml.load(f)

    assert config_toml["dataset"]["dataset"], "dataset.dataset cannot be None."
    assert config_toml["model"]["net"], "model.net cannot be None."

    config_enumerated = enumerate_config(config_toml)

    config_args = [Config(config) for config in config_enumerated]

    return config_enumerated, config_args

def enumerate_config(config):
    
    dict_stack = [config]
    dict_results = []

    # Generate different configs from lists
    while dict_stack:
        config_dict = dict_stack.pop()
        has_no_lists = True
        for top_key, sub_dict in config_dict.items():
            for sub_key, value in sub_dict.items():
                if isinstance(value, (list, tuple)) and len(value) > 1:
                    has_no_lists = False 
                    print(f"{sub_key} has more than one elements")

                    for elem in value:
                        config_copy = copy.deepcopy(config_dict)
                        config_copy[top_key][sub_key] = elem
                        dict_stack.append(config_copy)
                if not has_no_lists:
                    break
            if not has_no_lists:
                break
        
        if has_no_lists:
            dict_results.append(config_dict)

    # Fix one element list and tuples
    for sub_config in dict_results:
        for sub_dict in sub_config.values():
            for key, value in sub_dict.items():
                if isinstance(value, (list, tuple)):
                    sub_dict[key] = value[0]    
    
    return dict_results

if __name__ == "__main__":
    config_dict_list, config_args_list = load_config("src/experiments/exp1_config.toml")

    for config_dict, args in zip(config_dict_list, config_args_list):
        print("Running experiment with following config:")
        pprint(config_dict)
        experiment_run(args)
