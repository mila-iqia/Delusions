"""
MAIN MULTIPROCESSED SCRIPT FOR RUNNING SKIPPER TRAINING
"""

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch.multiprocessing as multiprocessing
from utils import *
from runtime import generate_exptag, get_set_seed, config_parser
import utils_mp

if __name__ == "__main__":
    parser = config_parser(mp=True)
    args = parser.parse_args()

    config_train = {"size": args.size_world, "gamma": args.gamma, "lava_density_range": [0.4, 0.4], "uniform_init": bool(args.uniform_init), "stochasticity": args.stochasticity, "singleton": not bool(args.nonsingleton)}

    configs_eval = [
        {
            "size": args.size_world,
            "gamma": args.gamma,
            "lava_density_range": [0.2, 0.3],
            "uniform_init": False,
            "stochasticity": args.stochasticity,
            "singleton": not bool(args.nonsingleton),
        },
        {
            "size": args.size_world,
            "gamma": args.gamma,
            "lava_density_range": [0.3, 0.4],
            "uniform_init": False,
            "stochasticity": args.stochasticity,
            "singleton": not bool(args.nonsingleton),
        },
        {
            "size": args.size_world,
            "gamma": args.gamma,
            "lava_density_range": [0.4, 0.5],
            "uniform_init": False,
            "stochasticity": args.stochasticity,
            "singleton": not bool(args.nonsingleton),
        },
        {
            "size": args.size_world,
            "gamma": args.gamma,
            "lava_density_range": [0.5, 0.6],
            "uniform_init": False,
            "stochasticity": args.stochasticity,
            "singleton": not bool(args.nonsingleton),
        },
    ]

    if args.game == "RandDistShift":
        from runtime import get_new_env_RDS
        func_get_new_env = get_new_env_RDS
    elif args.game == "SwordShieldMonster":
        from runtime import get_new_env_SSM
        func_get_new_env = get_new_env_SSM
    else:
        raise NotImplementedError("what is this game?")

    env = func_get_new_env(args, **config_train)
    args = generate_exptag(args, additional="")
    args.seed = get_set_seed(args.seed, env)

    print(args)

    # MAIN
    multiprocessing.set_start_method("spawn")
    utils_mp.run_multiprocess(args, config_train, configs_eval)
