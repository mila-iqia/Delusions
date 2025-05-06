import os, json
import torch
import time, datetime, numpy as np, copy
from tqdm import tqdm
from gym.envs.registration import register as gym_register

gym_register(id="RandDistShift-v2", entry_point="RandDistShift:RandDistShift2", reward_threshold=0.95)
gym_register(id="SwordShieldMonster-v2", entry_point="SwordShieldMonster:SwordShieldMonster2", reward_threshold=0.95)

from tensorboardX import SummaryWriter
from runtime import generate_exptag, get_set_seed, evaluate_agent, config_parser
from utils import get_cpprb_env_dict, evaluate_multihead_minigrid, minigridobs2tensor, process_batch

from models import CVAE_MiniGrid
from common.HER import HindsightReplayBuffer

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
    raise NotImplementedError("this script is for either RandDistShift or SwordShieldMonster")

if args.num_envs_train > 0:
    envs_train = []
    for idx_env in tqdm(range(args.num_envs_train), desc="create and solve envs"):
        env = func_get_new_env(args, **config_train)
        env.reset()
        env.generate_oracle()
        env.generate_obses_all()
        env.DP_info["obses_all_processed"] = minigridobs2tensor(env.DP_info["obses_all"], device=device)
        env.DP_info["i_all"], env.DP_info["j_all"], env.DP_info["x_all"] = env.obs2ijxd(env.DP_info["obses_all"])
        env.DP_info["omega_all_states_existent"] = torch.tensor(env.DP_info["omega_states"][env.DP_info["states_reachable"]], device=device)
        env.DP_info["Q_optimal_existent"] = torch.tensor(env.DP_info["Q_optimal"][env.DP_info["states_reachable"]], device=device)
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
    raise NotImplementedError("not for this script")
elif args.method == "DQN":
    from baselines import create_DQN_agent
    agent = create_DQN_agent(args, env=env, dim_embed=args.dim_embed, num_actions=env.action_space.n)
elif args.method == "random":
    from baselines import create_RW_agent
    agent = create_RW_agent(args, env=env, dim_embed=args.dim_embed, num_actions=env.action_space.n)
else:
    raise NotImplementedError("what is this agent?")

################################################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#########################################
num_categoricals, num_categories = 8, 2
argmax_reconstruction = True
args.disable_eval = 1
#########################################
depth, width = 2, 256
atoms = 4
to_uniform = False
debug = True
prioritized_cvae = True
freq_visualize_generation = 10000
eps_adam = 1.5e-4
size_batch_cvae = args.size_batch
onehot_state = False
activation = torch.nn.ReLU
additional_goals = 4
unique_goals = False
local_comments = f""

if onehot_state:
    local_comments += "_onehot"
else:
    local_comments += "_compact"
local_comments += "_unlimited_CVAE_buffer"
if prioritized_cvae:
    local_comments += "_prior"
else:
    local_comments += "_noprior"
if eps_adam != 1.5e-4:
    local_comments += f"_eps{eps_adam}"

if to_uniform:
    local_comments += "_to_uniform"

if size_batch_cvae != args.size_batch:
    local_comments += f"_bs_cvae{size_batch_cvae:d}"

if unique_goals:
    local_comments += "_unique_goals"

while len(local_comments) and local_comments[0] == "_":
    local_comments = local_comments[1:]
while len(local_comments) and local_comments[-1] == "_":
    local_comments = local_comments[:-1]

env_dict = get_cpprb_env_dict(env)
hrb = HindsightReplayBuffer(
    additional_goals * args.size_buffer,
    env_dict,
    max_episode_len=env.unwrapped.max_steps,
    reward_func=None,
    prioritized=prioritized_cvae,
    strategy=args.hindsight_strategy,
    additional_goals=additional_goals,
    num_goals_per_transition=1,
    unique_goals=unique_goals,
    num_envs=args.num_envs_train,
)

obs2tensor = lambda obs: minigridobs2tensor(obs, device=DEVICE)

if args.game == "RandDistShift":
    from models import Encoder_MiniGrid_RDS, Decoder_MiniGrid_RDS
    layout_extractor = Encoder_MiniGrid_RDS()
    decoder = Decoder_MiniGrid_RDS()
    num_classes_abstract = 1
else:
    from models import Encoder_MiniGrid_SSM, Decoder_MiniGrid_SSM
    layout_extractor = Encoder_MiniGrid_SSM()
    decoder = Decoder_MiniGrid_SSM()
    num_classes_abstract = 4

# NOTE: can use the following 2 lines to make sure that with the agent_ijx_mask is consistently extracted and used
# for _ in tqdm(range(100)):
# env = func_get_new_env(args, **config_train)
obs_sample = obs2tensor(env.reset())
sample_layout, sample_mask_agent = layout_extractor(obs_sample)
obs_recon = decoder(sample_layout, sample_mask_agent)
assert (obs_sample == obs_recon).all()

cvae = CVAE_MiniGrid(
    layout_extractor,
    decoder,
    obs2tensor(env.reset()),
    num_categoricals=num_categoricals,
    num_categories=num_categories,
    maximize_entropy=to_uniform,
    activation=activation,
    num_classes_abstract=num_classes_abstract,
    argmax_reconstruction=argmax_reconstruction,
)

cvae.to(DEVICE)
params_cvae = cvae.parameters()
optimizer_cvae = torch.optim.Adam(params_cvae, lr=args.lr, eps=eps_adam)

milestones_evaluation = []
step_milestone, pointer_milestone = 0, 0
while step_milestone <= args.steps_stop:
    milestones_evaluation.append(step_milestone)
    step_milestone += args.freq_eval

path_writer = f"tb_records/{env.spec.id}/{args.size_world}x{args.size_world}_traindiff{args.difficulty:g}"
if args.num_envs_train > 0:
    path_writer += f"x{args.num_envs_train:d}"

path_writer += f"_CVAE/{args.method}/{args.comments}_CVAE_{num_categoricals}x{num_categories}_depth{depth}_width{width}_atoms{atoms}_{local_comments}/{args.seed}"
writer = SummaryWriter(path_writer)
with open(os.path.join(path_writer, 'args.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

episode_elapsed, step_last_eval = 0, -freq_visualize_generation
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
        real_done = done and not info["overtime"]
        if agent.steps_interact - step_last_eval >= freq_visualize_generation and not real_done:
            idx_config = np.random.choice(range(len(configs_eval)))
            config_eval = configs_eval[idx_config]
            env_debug = func_get_new_env(args, **config_eval)
            obs_cond = env_debug.reset()
            step_last_eval += freq_visualize_generation
        sample = {"obs": obs_curr, "act": action, "rew": reward, "next_obs": obs_next, "done": real_done, "idx_env": env.idx_env}
        hrb.add(**sample)
        if agent.steps_interact >= agent.time_learning_starts and hrb.get_stored_size() > size_batch_cvae and agent.steps_interact % 4 == 0:
            flag_debug = debug and agent.steps_interact % 100 == 0
            batch = hrb.sample(size_batch_cvae)
            batch_processed = process_batch(batch, prioritized=prioritized_cvae, with_targ=True, obs2tensor=minigridobs2tensor, device=DEVICE)
            cvae.train()
            (
                loss_overall,
                loss_recon,
                loss_entropy,
                loss_conditional_prior,
                loss_align,
                dist_L1_mean,
                dist_L1_nontrivial,
                dist_L1_trivial,
                uniformity,
                entropy_prior,
                ratio_perfect_recon,
                ratio_aligned,
            ) = cvae.compute_loss(batch_processed, debug=flag_debug)
            if prioritized_cvae:
                weights_rb, idxes_rb = batch_processed[-2], batch["indexes"]
                loss_overall_weighted = (loss_overall * weights_rb.detach().squeeze()).mean()
            else:
                loss_overall_weighted = loss_overall.mean()
            optimizer_cvae.zero_grad(set_to_none=True)
            loss_overall_weighted.backward()
            torch.nn.utils.clip_grad_value_(params_cvae, 1.0)
            optimizer_cvae.step()
            with torch.no_grad():
                if prioritized_cvae:
                    if loss_align is not None:
                        loss_align_weighted = (loss_align * weights_rb.detach().squeeze()).mean()
                    loss_entropy_weighted = (loss_entropy * weights_rb.detach().squeeze()).mean()
                    loss_recon_weighted = (loss_recon * weights_rb.detach().squeeze()).mean()
                    if loss_conditional_prior is not None:
                        loss_conditional_prior_weighted = (loss_conditional_prior * weights_rb.detach().squeeze()).mean()
                else:
                    if loss_align is not None:
                        loss_align_weighted = loss_align.mean()
                    loss_entropy_weighted = loss_entropy.mean()
                    loss_recon_weighted = loss_recon.mean()
                    if loss_conditional_prior is not None:
                        loss_conditional_prior_weighted = loss_conditional_prior.mean()
                if prioritized_cvae:
                    hrb.update_priorities(idxes_rb, loss_overall.detach().cpu().numpy().squeeze())

            writer.add_scalar(f"Loss/recon", loss_recon_weighted.item(), agent.steps_interact)
            writer.add_scalar(f"Loss/entropy", loss_entropy_weighted.item(), agent.steps_interact)
            if loss_conditional_prior is not None:
                writer.add_scalar(f"Loss/conditional_prior", loss_conditional_prior_weighted.item(), agent.steps_interact)
            if loss_align is not None:
                writer.add_scalar(f"Loss/align", loss_align_weighted.item(), agent.steps_interact)
            writer.add_scalar(f"Loss/overall", loss_overall_weighted.item(), agent.steps_interact)
            if debug and agent.steps_interact % 100 == 0:
                writer.add_scalar(f"Dist/L1", dist_L1_mean.item(), agent.steps_interact)
                writer.add_scalar(f"Dist/L1_nontrivial", dist_L1_nontrivial.item(), agent.steps_interact)
                writer.add_scalar(f"Dist/L1_trivial", dist_L1_trivial.item(), agent.steps_interact)
                if entropy_prior is not None:
                    writer.add_scalar(f"Dist/entropy_prior", entropy_prior.item(), agent.steps_interact)
                writer.add_scalar(f"Dist/ratio_imperfect_recon", 1 - ratio_perfect_recon.item(), agent.steps_interact)
                writer.add_scalar(f"Dist/ratio_unaligned", 1 - ratio_aligned.item(), agent.steps_interact)
        steps_episode += 1
        agent.step(obs_curr, action, reward, obs_next, done and not info["overtime"], writer=writer)
        return_cum += reward
        return_cum_discount += reward * args.gamma ** env.step_count
        obs_curr = obs_next
    if done:
        agent.on_episode_end()
        hrb.on_episode_end()
        time_episode_end = time.time()
        if not args.random_walk:
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

        if args.method in ["random", "leap"]:
            epsilon = 1.0
        else:
            epsilon = agent.schedule_epsilon.value(max(0, agent.steps_interact))
        str_info += (
            f"seed: {args.seed}, steps_interact: {agent.steps_interact}, episode: {episode_elapsed}, "
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
env.close()
time_duration = time_end - time_start
print("total time elapsed: %s" % str(datetime.timedelta(seconds=time_duration)))
