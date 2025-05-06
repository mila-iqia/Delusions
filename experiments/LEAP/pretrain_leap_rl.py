import os
import time, datetime, numpy as np, copy
from tqdm import tqdm
from gym.envs.registration import register as gym_register

gym_register(id="RandDistShift-v2", entry_point="RandDistShift:RandDistShift2", reward_threshold=0.95)
gym_register(id="SwordShieldMonster-v2", entry_point="SwordShieldMonster:SwordShieldMonster2", reward_threshold=0.95)

from tensorboardX import SummaryWriter
from runtime import generate_exptag, get_set_seed, config_parser, save_code_snapshot, evaluate_agent

from utils import evaluate_multihead_minigrid_LEAP, minigridobs2tensor, decipher_hindsight_strategies


parser = config_parser(mp=False)
args = parser.parse_args()
assert args.method == "LEAP"

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_train = {
    "size": args.size_world,
    "gamma": args.gamma,
    "lava_density_range": [args.difficulty, args.difficulty],
    "uniform_init": bool(args.uniform_init),
    "stochasticity": args.stochasticity,
}

configs_eval = [
    {"size": args.size_world, "gamma": args.gamma, "lava_density_range": [0.2, 0.3], "uniform_init": False, "stochasticity": args.stochasticity},
    {"size": args.size_world, "gamma": args.gamma, "lava_density_range": [0.3, 0.4], "uniform_init": False, "stochasticity": args.stochasticity},
    {"size": args.size_world, "gamma": args.gamma, "lava_density_range": [0.4, 0.5], "uniform_init": False, "stochasticity": args.stochasticity},
    {"size": args.size_world, "gamma": args.gamma, "lava_density_range": [0.5, 0.6], "uniform_init": False, "stochasticity": args.stochasticity}
]

if args.game == "RandDistShift":
    from runtime import get_new_env_RDS
    func_get_new_env = get_new_env_RDS
elif args.game == "SwordShieldMonster":
    from runtime import get_new_env_SSM
    func_get_new_env = get_new_env_SSM
else:
    raise NotImplementedError("what is this game?")

envs_train = []

env = func_get_new_env(args, **config_train)

args.seed_rl_run = np.random.randint(0, 1000000)
assert len(args.seed), "must load vae checkpoint"
args.seed = get_set_seed(args.seed, env)

args.num_waypoints = 5

args = generate_exptag(args, additional="")
args.random_walk_leap = True
args.random_walk = True
hindsight_strategy_primary, hindsight_strategy_secondary, pertask_mixrate = decipher_hindsight_strategies(args.hindsight_strategy)
path_base = f"tb_records/{env.spec.id}/{args.size_world}x{args.size_world}_traindiff{args.difficulty:g}"
if args.num_envs_train > 0:
    path_base += f"x{args.num_envs_train:d}"
path_base += f"/{args.method}/"
path_saved = path_base + f"vae_pretrain/{hindsight_strategy_primary}/{args.seed}"
args.path_pretrained_vae = os.path.join(path_saved, "cvae.pt")
assert os.path.exists(args.path_pretrained_vae)
args.path_pretrain_envs = os.path.join(path_saved, "envs.pkl")
assert os.path.exists(args.path_pretrain_envs)

path_writer = path_base + f"rl_train/{args.comments}/from{args.seed}/{args.seed_rl_run}"

writer = SummaryWriter(path_writer)

if args.num_envs_train:
    with open(args.path_pretrain_envs, "rb") as file:
        import pickle
        envs_train = pickle.load(file)

    for idx_env in tqdm(range(args.num_envs_train), desc="create and solve envs"):
        env = envs_train[idx_env]
        env.reset()
        env.init_DP_assets()
        env.collect_transition_probs()
        env.collect_state_adjacency()
        env.generate_oracle()
        env.generate_obses_all()
        env.DP_info["obses_all_processed"] = minigridobs2tensor(env.DP_info["obses_all"], device=device)
        env.DP_info["omega_all_states_existent"] = torch.tensor(env.DP_info["omega_states"][env.DP_info["states_reachable"]], device=device)
        env.DP_info["Q_optimal_existent"] = torch.tensor(env.DP_info["Q_optimal"][env.DP_info["states_reachable"]], device=device)
        env.idx_env = idx_env
    def generator_env_train():
        idx_env = np.random.randint(args.num_envs_train)
        return copy.copy(envs_train[idx_env])
else:
    def generator_env_train():
        env_train = func_get_new_env(args, **config_train)
        return env_train


save_code_snapshot(path_writer)

print(args)

from LEAP import create_LEAP_agent
agent = create_LEAP_agent(args, env=env, dim_embed=args.dim_embed, num_actions=env.action_space.n)

milestones_evaluation = []
step_milestone, pointer_milestone = 0, 0
while step_milestone <= args.steps_stop:
    milestones_evaluation.append(step_milestone)
    step_milestone += args.freq_eval

episode_elapsed = 0
time_start = time.time()
return_cum, return_cum_discount, steps_episode, time_episode_start, str_info = 0.0, 0.0, 0, time.time(), ""

while True:
    if args.randomized:
        env = generator_env_train()
    obs_curr, done = env.reset(same_init_pos=False), False
    if not args.disable_eval and pointer_milestone < len(milestones_evaluation) and agent.steps_interact >= milestones_evaluation[pointer_milestone]:
        env_generator = lambda: func_get_new_env(args, **config_train) if args.randomized else None
        evaluate_multihead_minigrid_LEAP(env, agent, writer, size_batch=32, num_episodes=5, suffix="", step_record=None, env_generator=env_generator, queue_envs=None)
        env_generator = lambda: generator_env_train()
        returns_mean, returns_std, returns_discounted_mean, returns_discounted_std = evaluate_agent(env_generator, agent, num_episodes=20, type_env="minigrid")
        print(
            f"Eval/train x{20} @ step {agent.steps_interact:d} - returns_mean: {returns_mean:.2f}, returns_std: {returns_std:.2f}, returns_discounted_mean: {returns_discounted_mean:.2f}, returns_discounted_std: {returns_discounted_std:.2f}"
        )
        writer.add_scalar("Eval/train", returns_mean, agent.steps_interact)
        writer.add_scalar("Eval/train_discount", returns_discounted_mean, agent.steps_interact)
        for config_eval in configs_eval:
            env_generator = lambda: func_get_new_env(args, **config_eval)
            returns_mean, returns_std, returns_discounted_mean, returns_discounted_std = evaluate_agent(
                env_generator, agent, num_episodes=20, type_env="minigrid"
            )
            diff = np.mean(config_eval["lava_density_range"])
            print(
                f"Eval/{diff:g} x{20} @ step {agent.steps_interact:d} - returns_mean: {returns_mean:.2f}, returns_std: {returns_std:.2f}, returns_discounted_mean: {returns_discounted_mean:.2f}, returns_discounted_std: {returns_discounted_std:.2f}"
            )
            writer.add_scalar(f"Eval/{diff:g}", returns_mean, agent.steps_interact)
            writer.add_scalar(f"Eval/discount_{diff:g}", returns_discounted_mean, agent.steps_interact)
        pointer_milestone += 1
    if not (agent.steps_interact <= args.steps_max and episode_elapsed <= args.episodes_max and agent.steps_interact <= args.steps_stop):
        break
    while not done and agent.steps_interact <= args.steps_max:
        action = agent.decide(obs_curr, env=env, writer=writer, random_walk=bool(args.random_walk_leap))
        obs_next, reward, done, info = env.step(action)
        real_done = done and not info["overtime"]
        steps_episode += 1
        agent.step(obs_curr, action, reward, obs_next, real_done, idx_env=env.idx_env, writer=writer)
        return_cum += reward
        return_cum_discount += reward * args.gamma ** env.step_count
        obs_curr = obs_next
    if done:
        agent.on_episode_end()
        time_episode_end = time.time()
        writer.add_scalar("Experience/return", return_cum, agent.steps_interact)
        writer.add_scalar("Experience/return_discount", return_cum_discount, agent.steps_interact)
        if args.game == "RandDistShift":
            writer.add_scalar("Experience/dist2init", info["dist2init"], agent.steps_interact)
            writer.add_scalar("Experience/dist2goal", info["dist2goal"], agent.steps_interact)
            writer.add_scalar("Experience/dist2init_x", np.abs(info["agent_pos"][0] - info["agent_pos_init"][0]), agent.steps_interact)
        elif args.game == "SwordShieldMonster":
            writer.add_scalar("Experience/sword_acquired", float(info["sword_acquired"]), agent.steps_interact)
            writer.add_scalar("Experience/shield_acquired", float(info["shield_acquired"]), agent.steps_interact)
        writer.add_scalar("Experience/overtime", float(info["overtime"]), agent.steps_interact)
        writer.add_scalar("Experience/episodes", episode_elapsed, agent.steps_interact)

        if args.method in ["random"]:
            epsilon = 1.0
        else:
            epsilon = agent.schedule_epsilon.value(max(0, agent.steps_interact))
        str_info += (
            f"seed_rl_run: {args.seed_rl_run}, steps_interact: {agent.steps_interact}, episode: {episode_elapsed}, "
            f"epsilon: {epsilon: .2f}, return: {return_cum: g}, return_discount: {return_cum_discount: g}, "
            f"steps_episode: {steps_episode}"
        )
        duration_episode = time_episode_end - time_episode_start
        if duration_episode and agent.steps_interact >= agent.time_learning_starts:
            sps_episode = steps_episode / duration_episode
            writer.add_scalar("Other/sps", sps_episode, agent.steps_interact)
            eta = str(datetime.timedelta(seconds=int((args.steps_stop - agent.steps_interact) / sps_episode)))
            str_info += ", sps_episode: %.2f, eta: %s" % (sps_episode, eta)
        print(str_info)
        writer.add_text("Text/info_train", str_info, agent.steps_interact)
        return_cum, return_cum_discount, steps_episode, time_episode_start, str_info = (0, 0, 0, time.time(), "")
        episode_elapsed += 1
time_end = time.time()
time_duration = time_end - time_start
print("total time elapsed: %s" % str(datetime.timedelta(seconds=time_duration)))
