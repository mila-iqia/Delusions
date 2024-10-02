import os
import random, numpy as np, torch, time, argparse
import gym


def get_set_seed(seed, env):
    if len(seed):
        seed = int(seed)
    else:
        seed = random.randint(0, 1000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        env.seed(seed)
    except:
        print("failed to set env seed")
    return seed


def generate_exptag(args, additional=""):
    if args.comments == "x":
        args.comments = ""
    if len(additional):
        args.comments += additional
    if args.activation != "relu":
        args.comments += "_%s" % (args.activation)
    if not args.prioritized_replay:
        args.comments += "_noprior"
    if args.no_replay_rewrite:
        args.comments += "_no_replay_rewrite"
    if args.freq_train != 4:
        args.comments += "_freq_train%d" % (args.freq_train)
    if args.freq_targetsync != 8000:
        args.comments += "_targetsync%d" % (args.freq_targetsync)
    if "minigrid" in args.game.lower() or "randdistshift" in args.game.lower() or "swordshieldmonster" in args.game.lower():
        if args.stochasticity > 0.0:
            args.comments += "_stoch%.2f" % (args.stochasticity)
        if "leap" in args.method.lower():
            args.comments += f"{args.depth_hidden}x{args.width_hidden}_lenrep{args.len_rep}"
        else:
            args.comments += f"{args.depth_hidden}x{args.width_hidden}_{args.atoms_value}atoms_lenrep{args.len_rep}"
        if args.cvae:
            args.comments += "_CVAE"
            if args.prob_relabel_generateJIT:
                args.comments += f"_prob_relabel_generateJIT{args.prob_relabel_generateJIT:g}"
        if args.uniform_init:
            args.comments += "_uniform_init"
        if args.method == "DQN_Skipper":
            if args.transform_discount_target:
                args.comments += "_learn_dist"
            if args.no_Q_head:
                args.comments += "_no_Q"
            else:
                args.comments += f"_{args.type_intrinsic_reward}{args.gamma_int:.2f}"
        if args.append_pos:
            args.comments += "_append_pos"
        if "local" in args.arch_enc and args.method.lower() != "leap":
            args.comments += f"_{args.num_heads}heads_top{args.size_bottleneck:d}"
        if args.random_walk:
            args.comments += "_RW"
        if args.method == "DQN_Skipper":
            if args.always_select_goal:
                args.comments += "_always_select_goal"
            if args.optimal_plan:
                args.comments += "_optimal_plan"
            if args.optimal_policy:
                args.comments += "_optimal_policy"
        if args.clip_reward:
            args.comments += "_clip_reward"
        if args.size_batch != 64:
            args.comments += "_bs%d" % (args.size_batch)
        if args.lr != 0.00025:
            args.comments += "_lr_%gx" % (args.lr / 0.00025)
        if not args.randomized:
            args.comments += "_static"
        if args.method == "DQN_Skipper":
            args.comments += f"_{args.num_waypoints_unpruned:d}to{args.num_waypoints:d}wps_every{args.freq_plan:d}_{args.waypoint_strategy}"
        elif "leap" in args.method.lower():
            args.comments += f"_{args.num_waypoints:d}wps_every{args.freq_plan:d}"
        args.comments += f"_{args.hindsight_strategy}"
        if "leap" in args.method.lower():
            args.comments += f"_dim_latent{args.dim_latent_leap:d}"
        else:
            args.comments += f"_lenebd{args.dim_embed}"
            args.comments += f"_{args.arch_enc}"
            if args.method == "DQN_Skipper":
                if not args.cvae:
                    if args.valid_waypoints_only:
                        if args.no_lava_waypoints:
                            args.comments += "_valid_nonlava_wps"
                        else:
                            args.comments += "_valid_wps"
                    elif args.no_lava_waypoints:
                        args.comments += "_nonlava_wps"
                if args.unique_codes:
                    args.comments += "_unique_codes"
                if args.unique_obses:
                    args.comments += "_unique_obses"
    elif "atari" in args.game.lower():
        if args.clip_reward:
            args.comments += "_clip"
        if args.size_batch != 32:
            args.comments += "_bs%d" % (args.size_batch)
        if not args.framestack:
            args.comments += "_nostack"
        if args.lr != 0.0000625:
            args.comments += "_lr_%gx" % (args.lr / 0.0000625)
    elif "procgen" in args.game.lower():
        if args.clip_reward:
            args.comments += "_clip"
        if args.size_batch != 512:
            args.comments += "_bs%d" % (args.size_batch)
        if args.framestack:
            args.comments += "_stack%d" % (args.framestack)
        if args.lr != 0.00025:
            args.comments += "_lr_%gx" % (args.lr / 0.00025)
    if args.gamma != 0.99:
        args.comments += "_%.2f" % (args.gamma)
    if not args.layernorm:
        args.comments += "_nonorm"
    while args.comments[0] == "_":
        args.comments = args.comments[1:]
    return args


def get_new_env_RDS(args, size=8, lava_density_range=[0.3, 0.4], min_num_route=1, uniform_init=False, gamma=0.99, stochasticity=0.0):
    env = gym.make(
        "RandDistShift-%s" % args.version_game,
        width=size,
        height=size,
        lava_density_range=lava_density_range,
        min_num_route=min_num_route,
        ignore_color=False,
        uniform_init=uniform_init,
        gamma=gamma,
        stochasticity=stochasticity,
    )
    return env

def get_new_env_SSM(args, size=8, lava_density_range=[0.3, 0.4], uniform_init=False, gamma=0.99, stochasticity=0.0):
    env = gym.make(
        "SwordShieldMonster-v2",
        width=size,
        height=size,
        lava_density_range=lava_density_range,
        uniform_init=uniform_init,
        gamma=gamma,
        stochasticity=stochasticity,
    )
    return env


@torch.no_grad()
def evaluate_agent(func_env, agent, num_episodes=5, type_env="minigrid", queue_envs=None, writer=None):
    returns, returns_discounted = [], []
    agent.on_episode_end(eval=True)
    for _ in range(num_episodes):
        if queue_envs is not None:
            env = None
            while env is None:
                try:
                    env = queue_envs.get()
                except:
                    time.sleep(0.001)
        else:
            env = func_env()
        obs_curr, done, flag_reset = env.reset(same_init_pos=False), False, False
        steps_episode, return_episode, return_episode_discounted = 0, 0, 0
        while not flag_reset:
            action = agent.decide(obs_curr, env=env, eval=True, writer=writer, random_walk=False)
            obs_next, reward, done, info = env.step(action)  # take a computed action
            steps_episode += 1
            return_episode += reward
            return_episode_discounted += reward * agent.gamma**env.step_count
            obs_curr = obs_next
            if type_env == "procgen":
                flag_reset = done and steps_episode != env.spec.max_episode_steps and reward == 0 and not info["prev_level_complete"]
            elif type_env == "atari":
                flag_reset = env.was_real_done
            else:
                flag_reset = done
        agent.on_episode_end(eval=True)
        returns.append(np.copy(return_episode))
        returns_discounted.append(np.copy(return_episode_discounted))
    returns_mean, returns_std = np.mean(returns), np.std(returns)
    returns_discounted_mean, returns_discounted_std = np.mean(returns_discounted), np.std(returns_discounted)
    return returns_mean, returns_std, returns_discounted_mean, returns_discounted_std

def config_parser(mp=True):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--method", type=str, default="DQN_Skipper", help="type of agent")
    parser.add_argument("--game", type=str, default="SwordShieldMonster", choices=["RandDistShift", "SwordShieldMonster"])
    parser.add_argument(
        "--version_game", type=str, default="v2", choices=["v2"], help="v1 (turn-OR-forward), v2 (directional forward) or v3 (turn-AND-forward)"
    )
    parser.add_argument("--size_world", type=int, default=12, help="the length of each dimension for gridworlds")
    parser.add_argument('--difficulty', type=float, default=0.4, help='')
    parser.add_argument("--stochasticity", type=float, default=0.0, help="probability of random actions")
    parser.add_argument("--cvae", type=int, default=1, help="0 for using oracle future states, 1 for cvae generated")
    parser.add_argument("--randomized", type=int, default=1, help="")
    parser.add_argument("--uniform_init", type=int, default=1, help="uniform init for training")
    parser.add_argument("--num_envs_train", type=int, default=50, help="0 for inf")
    parser.add_argument("--seed", type=str, default="", help="if not set manually, would be random")
    parser.add_argument("--steps_stop", type=int, default=1500000, help="#agent-environment interactions before the experiment stops")
    parser.add_argument("--freq_eval", type=int, default=50000, help="interval of periodic evaluation (steps)")
    # arguments for agent setting
    parser.add_argument('--dist_out', type=int, default=1, help='')
    parser.add_argument('--num_categoricals', type=int, default=2, help='')
    parser.add_argument('--num_categories', type=int, default=2, help='')
    parser.add_argument('--argmax_latents', type=int, default=0, help='')
    parser.add_argument('--situations_share_encoder', type=int, default=1, help='')
    parser.add_argument("--freq_train", type=int, default=4, help="train every this number of steps")
    parser.add_argument("--freq_targetsync", type=int, default=8000, help="sync the target network every this number of steps")
    parser.add_argument("--random_walk", type=int, default=0, help="")
    parser.add_argument("--random_walk_leap", type=int, default=1, help="")
    parser.add_argument("--always_select_goal", type=int, default=0, help="")
    parser.add_argument("--optimal_plan", type=int, default=0, help="")
    parser.add_argument("--optimal_policy", type=int, default=0, help="")
    parser.add_argument("--prioritized_replay", type=int, default=1, help="prioritized replay buffer, good stuff!")
    parser.add_argument("--no_replay_rewrite", type=int, default=0, help="")
    parser.add_argument("--freq_plan", type=int, default=8, help="")
    parser.add_argument("--num_waypoints", type=int, default=24, help="")
    parser.add_argument("--num_waypoints_unpruned", type=int, default=64, help="")
    parser.add_argument("--valid_waypoints_only", type=int, default=1, help="")
    parser.add_argument("--no_lava_waypoints", type=int, default=1, help="")
    parser.add_argument("--waypoint_strategy", type=str, default="once", choices=["once", "regenerate_whole_graph"], help="")
    parser.add_argument("--hindsight_strategy", type=str, default="future+episode@0.5", help="")
    parser.add_argument("--layernorm", type=int, default=1, help="")
    parser.add_argument("--transform_discount_target", type=int, default=1, help="")
    parser.add_argument("--append_pos", type=int, default=0, help="")
    parser.add_argument("--num_heads", type=int, default=1, help="")
    parser.add_argument("--size_bottleneck", type=int, default=4, help="")
    parser.add_argument("--aux_share_encoder", type=int, default=1, help="")
    parser.add_argument("--prob_relabel_generateJIT", type=float, default=0.0, help="probability to relabel obs_targ2 with JIT generated obses")
    parser.add_argument("--unique_codes", type=int, default=0, help="prune the graph to only contain unique codes")
    parser.add_argument("--unique_obses", type=int, default=1, help="prune the graph to only contain unique obses")
    # arguments that shouldn't be changed
    parser.add_argument("--lr", type=float, default=0.00025, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount")
    parser.add_argument("--gamma_int", type=float, default=0.95, help="intrinsic discount for policy")
    parser.add_argument("--type_intrinsic_reward", type=str, default="sparse", choices=["sparse", "dense"])
    parser.add_argument("--depth_hidden", type=int, default=3, help="")
    parser.add_argument("--width_hidden", type=int, default=256, help="")
    parser.add_argument("--dim_embed", type=int, default=16, help="")
    parser.add_argument("--arch_enc", type=str, default="local", choices=["flatten", "local"], help="")
    parser.add_argument("--len_rep", type=int, default=128, help="")
    parser.add_argument("--size_buffer", type=int, default=1000000, help="size of replay buffer")
    parser.add_argument("--size_batch", type=int, default=64, help="batch size for training")
    parser.add_argument("--steps_max", type=int, default=50000000, help="set to be 50M for DQN to perform normally, since exploration period is a percentage")
    parser.add_argument("--episodes_max", type=int, default=50000000, help="a criterion just in case we need it")
    parser.add_argument("--no_Q_head", type=int, default=0, help="if no_Q_head, use shortest distance to guide agent")
    parser.add_argument("--atoms_value", type=int, default=16, help="for value estimator categorical output")
    parser.add_argument("--atoms_reward", type=int, default=16, help="for atoms_reward estimator categorical output")
    parser.add_argument("--atoms_discount", type=int, default=16, help="for atoms_discount estimator categorical output")
    parser.add_argument("--value_min", type=float, default=0.0, help="lower boundary for value estimator output")
    parser.add_argument("--value_max", type=float, default=1.0, help="upper boundary for value estimator output")
    parser.add_argument("--clip_reward", type=int, default=0, help="clip the reward to sign(reward) as in DQN")
    parser.add_argument("--activation", type=str, default="relu", help="")
    # arguments for runtime
    parser.add_argument("--disable_eval", type=int, default=0, help="")
    parser.add_argument("--force_cpu", type=int, default=0, help="")
    # for leap only
    parser.add_argument("--dim_latent_leap", type=int, default=64, help="")

    # arguments for identification
    if mp:
        parser.add_argument("--num_explorers", type=int, default=8, help="")
        parser.add_argument(
            "--comments",
            type=str,
            default="",
            help="If changed, the run will be marked with the string",
        )
    else:
        parser.add_argument("--comments", type=str, default="sp", help="If changed, the run will be marked with the string")
    return parser


def save_code_snapshot(path_tf_events):
    import zipfile
    from pathlib import Path

    def zipdir(path, ziph):
        # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            if ".git" in dirs:
                dirs.remove(".git")
            if ".ipynb_checkpoints" in dirs:
                dirs.remove(".ipynb_checkpoints")
            if ".github" in dirs:
                dirs.remove(".github")
            if ".vscode" in dirs:
                dirs.remove(".vscode")
            if "results" in dirs:
                dirs.remove("results")
            if "tb_records" in dirs:
                dirs.remove("tb_records")
            if "build" in dirs:
                dirs.remove("build")
            if "__pycache__" in dirs:
                dirs.remove("__pycache__")
            if "BACKUP" in dirs:
                dirs.remove("BACKUP")
            for file in files:
                ziph.write(
                    os.path.join(root, file),
                    os.path.relpath(os.path.join(root, file), os.path.join(path, "..")),
                )

    source_file_name = os.path.join(path_tf_events, "source.zip")
    with zipfile.ZipFile(source_file_name, "w", zipfile.ZIP_LZMA) as zipf:
        zipdir(Path.cwd(), zipf)
