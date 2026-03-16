# Run grid search using a wandb sweep. Assigns wandb agents to GPUs and runs them in parallel
import os
from itertools import product
import argparse
import yaml


def gen_combinations(d):
    # https://stackoverflow.com/questions/50606454/cartesian-product-of-nested-dictionaries-of-lists
    keys, values = d.keys(), d.values()
    combinations = product(*values)

    for c in combinations:
        yield c[0]


def gen_dict_combinations(d):
    # https://stackoverflow.com/questions/50606454/cartesian-product-of-nested-dictionaries-of-lists
    keys, values = d.keys(), d.values()
    for c in product(*(gen_combinations(v) for v in values)):
        yield dict(zip(keys, c))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        required=True,
        help="Experiment name",
    )
    parser.add_argument(
        "-g",
        "--gpus",
        type=int,
        nargs="+",
        required=False,
        default=[0],
        help="Available GPUs to run this experimet on.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_dir = os.path.join("experiments", args.experiment)
    config_path = os.path.join(config_dir, "wandb_config.yaml")

    with open(config_path, "r") as f:
        config: dict = yaml.safe_load(f)

    scenarios = [c for c in gen_dict_combinations(config["parameters"])]

    # get set of avail gpus if not passed in

    # assign unique scenario idx to each setup
    __import__("ipdb").set_trace(context=3)
    scenario_idx = 0

    for scenario_idx, params in enumerate(scenarios):
        # assign scenarios to GPUs
        # gpu_idx =


    # for each seed




# # print experiment summary in a markdown-formatted table
# echo -e "\nExperiment summary\n"

# header_1="| Scenario Name | Alg | Env | Params |"
# header_2="| ----| ---- | ---- | ---- |"
# echo $header_1
# echo $header_2

# # print output to table
# table_line="| $scenario_name | $rl_alg | $env | $scenario_param | "
# echo $table_line

if __name__ == "__main__":
    main()
