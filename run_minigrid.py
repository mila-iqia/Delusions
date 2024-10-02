"""
MAIN SINGLE-PROCESSS SCRIPT FOR RUNNING SKIPPER TRAINING
EASIER FOR DEBUGGING
"""
import os, json


import torch
import time, datetime, numpy as np, copy
from tqdm import tqdm
from gym.envs.registration import register as gym_register

gym_register(id="RandDistShift-v2", entry_point="RandDistShift:RandDistShift2", reward_threshold=0.95)
gym_register(id="SwordShieldMonster-v2", entry_point="SwordShieldMonster:SwordShieldMonster2", reward_threshold=0.95)

from tensorboardX import SummaryWriter
from runtime import generate_exptag, get_set_seed, evaluate_agent, config_parser
from utils import get_cpprb, evaluate_multihead_minigrid, minigridobs2tensor

parser = config_parser(mp=False)
args = parser.parse_args()

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

if args.num_envs_train > 0:
    envs_train = []
    for idx_env in tqdm(range(args.num_envs_train)):
        env = func_get_new_env(args, **config_train)
        env.reset()
        env.generate_oracle(include_random=True)
        env.generate_obses_all()
        env.DP_info["obses_all_processed"] = minigridobs2tensor(env.DP_info["obses_all"], device=device)
        env.DP_info["omega_all_states_existent"] = torch.tensor(env.DP_info["omega_states"][env.DP_info["states_reachable"]], device=device)
        env.DP_info["Q_optimal_existent"] = torch.tensor(env.DP_info["Q_optimal"][env.DP_info["states_reachable"]], device=device)
        env.DP_info["Q_random_existent"] = torch.tensor(env.DP_info["Q_random"][env.DP_info["states_reachable"]], device=device)
        env.idx_env = idx_env
        envs_train.append(env)
    def generator_env_train():
        idx_env = np.random.randint(args.num_envs_train)
        return copy.copy(envs_train[idx_env])
else:
    def generator_env_train():
        env_train = func_get_new_env(args, **config_train)
        return env_train


env = func_get_new_env(args, **config_train)
args = generate_exptag(args, additional="")
args.seed = get_set_seed(args.seed, env)

print(args)

if args.method == "DQN_Skipper":
    from agents import create_DQN_Skipper_agent
    hrb = get_cpprb(env, args.size_buffer, prioritized=args.prioritized_replay, num_envs=args.num_envs_train, hindsight=True, hindsight_strategy=args.hindsight_strategy)
    agent = create_DQN_Skipper_agent(args, env=env, dim_embed=args.dim_embed, num_actions=env.action_space.n, hrb=hrb)
elif args.method == "DQN":
    from baselines import create_DQN_agent
    agent = create_DQN_agent(args, env=env, dim_embed=args.dim_embed, num_actions=env.action_space.n)
elif args.method == "DQN_AUX":
    from agents import create_DQN_AUX_agent
    agent = create_DQN_AUX_agent(args, env=env, dim_embed=args.dim_embed, num_actions=env.action_space.n)
elif args.method == "QRDQN":
    from baselines import create_QRDQN_agent
    agent = create_QRDQN_agent(args, env=env, dim_embed=args.dim_embed, num_actions=env.action_space.n)
elif args.method == "QRDQN_AUX":
    from agents import create_QRDQN_AUX_agent
    agent = create_QRDQN_AUX_agent(args, env=env, dim_embed=args.dim_embed, num_actions=env.action_space.n)
elif args.method == "IQN":
    from baselines import create_IQN_agent
    agent = create_IQN_agent(args, env=env, dim_embed=args.dim_embed, num_actions=env.action_space.n)
elif args.method == "IQN_AUX":
    from agents import create_IQN_AUX_agent
    agent = create_IQN_AUX_agent(args, env=env, dim_embed=args.dim_embed, num_actions=env.action_space.n)
else:
    raise NotImplementedError("what is this agent?")

milestones_evaluation = []
step_milestone, pointer_milestone = 0, 0
while step_milestone <= args.steps_stop:
    milestones_evaluation.append(step_milestone)
    step_milestone += args.freq_eval

path_writer = f"tb_records/{env.spec.id}/{args.size_world}x{args.size_world}_traindiff{args.difficulty:g}"
if args.num_envs_train > 0:
    path_writer += f"x{args.num_envs_train:d}"
path_writer += f"/{args.method}/{args.comments}/{args.seed}"

writer = SummaryWriter(path_writer)
with open(os.path.join(path_writer, 'args.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

episode_elapsed, step_last_eval = 0, 0
time_start = time.time()
return_cum, return_cum_discount, steps_episode, time_episode_start, str_info = 0.0, 0.0, 0, time.time(), ""

while True:
    if args.randomized:
        env = generator_env_train()
    obs_curr, done = env.reset(), False
    if not args.disable_eval and pointer_milestone < len(milestones_evaluation) and agent.steps_interact >= milestones_evaluation[pointer_milestone]:
        if args.method == "DQN_Skipper":
            env_generator = lambda: func_get_new_env(args, **config_train) if args.randomized else None
            evaluate_multihead_minigrid(env, agent, writer, size_batch=32, num_episodes=5, suffix="", step_record=None, env_generator=env_generator, queue_envs=None)
        env_generator = lambda: generator_env_train()
        returns_mean, returns_std, returns_discounted_mean, returns_discounted_std = evaluate_agent(env_generator, agent, num_episodes=20, type_env="minigrid")
        print(
            f"Eval/train x{20} @ step {agent.steps_interact:d} - returns_mean: {returns_mean:.2f}, returns_std: {returns_std:.2f}, returns_discounted_mean: {returns_discounted_mean:.2f}, returns_discounted_std: {returns_discounted_std:.2f}"
        )
        writer.add_scalar("Eval/train", returns_mean, agent.steps_interact)
        writer.add_scalar("Eval/train_discount", returns_discounted_mean, agent.steps_interact)
        for config_eval in configs_eval:
            env_generator = lambda: func_get_new_env(args, **config_eval)
            returns_mean, returns_std, returns_discounted_mean, returns_discounted_std = evaluate_agent(env_generator, agent, num_episodes=20, type_env="minigrid")
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
        if args.method == "random":
            action = env.action_space.sample()
        else:
            action = agent.decide(obs_curr, env=env, writer=writer)
        obs_next, reward, done, info = env.step(action)
        steps_episode += 1
        agent.step(obs_curr, action, reward, obs_next, done and not info["overtime"], idx_env=env.idx_env, writer=writer)
        return_cum += reward
        return_cum_discount += reward * args.gamma ** env.step_count
        obs_curr = obs_next
    if done:
        agent.on_episode_end()
        time_episode_end = time.time()
        debug = writer is not None and np.random.rand() < 0.05
        ##### DP part
        if debug:
            with torch.no_grad():
                Q_true_optimal = env.DP_info["Q_optimal_existent"]
                omega_all_states_existent = env.DP_info["omega_all_states_existent"]
                obses_all_states = env.DP_info["obses_all_processed"]
            if args.method != "DQN_Skipper":
                with torch.no_grad():
                    pred_Qs_all_states = agent.network_policy(obses_all_states, scalarize=True)
                    error_true_Q_optimal = torch.abs(pred_Qs_all_states - Q_true_optimal)
                    error_true_Q_optimal_nonterm = error_true_Q_optimal[~omega_all_states_existent]
                    writer.add_histogram("DP/res_Q_optimal_nonterm", error_true_Q_optimal_nonterm.squeeze().cpu().numpy(), agent.steps_interact)
                    writer.add_scalar("DP/res_Q_optimal_nonterm_avg", error_true_Q_optimal_nonterm.mean().item(), agent.steps_interact)
                    writer.add_scalar("DP/res_Q_optimal_nonterm_max", torch.max(error_true_Q_optimal_nonterm).item(), agent.steps_interact)
            if "AUX" in args.method:
                with torch.no_grad():
                    Q_true_random = env.DP_info["Q_random_existent"]
                    pred_Qs_all_states = agent.network_policy_aux(obses_all_states, scalarize=True)
                    error_true_Q_random = torch.abs(pred_Qs_all_states - Q_true_random)
                    error_true_Q_random_nonterm = error_true_Q_random[~omega_all_states_existent]
                writer.add_histogram("DP/res_Q_random_nonterm", error_true_Q_random_nonterm.squeeze().cpu().numpy(), agent.steps_interact)
                writer.add_scalar("DP/res_Q_random_nonterm_avg", error_true_Q_random_nonterm.mean().item(), agent.steps_interact)
                writer.add_scalar("DP/res_Q_random_nonterm_max", torch.max(error_true_Q_random_nonterm).item(), agent.steps_interact)
            ##### DP part
        if debug:
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

        epsilon = agent.schedule_epsilon.value(max(0, agent.steps_interact))
        str_info += (
            f"seed: {args.seed}, steps_interact: {agent.steps_interact}, episode: {episode_elapsed}, "
            f"epsilon: {epsilon: .2f}, return: {return_cum: g}, return_discount: {return_cum_discount: g}, "
            f"steps_episode: {steps_episode}"
        )
        duration_episode = time_episode_end - time_episode_start
        if duration_episode and agent.steps_interact >= agent.time_learning_starts:
            sps_episode = steps_episode / duration_episode
            if debug: writer.add_scalar("Other/sps", sps_episode, agent.steps_interact)
            eta = str(datetime.timedelta(seconds=int((args.steps_stop - agent.steps_interact) / sps_episode)))
            str_info += ", sps_episode: %.2f, eta: %s" % (sps_episode, eta)
        print(str_info)
        if debug: writer.add_text("Text/info_train", str_info, agent.steps_interact)
        return_cum, return_cum_discount, steps_episode, time_episode_start, str_info = (0, 0, 0, time.time(), "")
        episode_elapsed += 1
time_end = time.time()
env.close()
time_duration = time_end - time_start
print("total time elapsed: %s" % str(datetime.timedelta(seconds=time_duration)))
