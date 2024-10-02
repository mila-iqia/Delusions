"""
MAIN SCRIPT FOR RUNNING LEAP CVAE PRE-TRAINING
"""

import os, json

import time, datetime, numpy as np, copy, pickle
from tqdm import tqdm
from gym.envs.registration import register as gym_register

gym_register(id="RandDistShift-v2", entry_point="RandDistShift:RandDistShift2", reward_threshold=0.95)
gym_register(id="SwordShieldMonster-v2", entry_point="SwordShieldMonster:SwordShieldMonster2", reward_threshold=0.95)

from baselines import create_RW_agent

from tensorboardX import SummaryWriter
from runtime import generate_exptag, get_set_seed, config_parser
import torch

from utils import process_batch

from models import CVAE_MiniGrid
from utils import get_cpprb_env_dict, minigridobs2tensor
from HER import HindsightReplayBuffer

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import io
import cv2

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
    for idx_env in tqdm(range(args.num_envs_train), desc="create and solve envs"):
        env = func_get_new_env(args, **config_train)
        env.reset()
        env.idx_env = idx_env
        envs_train.append(env)
    def generator_env_train():
        idx_env = np.random.randint(args.num_envs_train)
        return copy.copy(envs_train[idx_env])
else:
    def generator_env_train():
        env_train = func_get_new_env(args, **config_train)
        return env_train



args.method = "LEAP"
args.comments = ""
env = func_get_new_env(args, **config_train)
args = generate_exptag(args, additional="")
args.seed = get_set_seed(args.seed, env)
args.steps_stop = 250000

print(args)

agent = create_RW_agent(args, env)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_categoricals, num_categories = 6, 2

argmax_reconstruction = False
args.disable_eval = 1
depth, width = 2, 256
prioritized_cvae = True
freq_visualize_generation = 10000
eps_adam = 1.5e-4
size_batch_cvae = args.size_batch
activation = torch.nn.ReLU
additional_goals = 4
local_comments = f"6x2_sampled_recon"

if prioritized_cvae:
    local_comments += "_prior"
else:
    local_comments += "_noprior"
if eps_adam != 1.5e-4:
    local_comments += f"_eps{eps_adam}"

if size_batch_cvae != args.size_batch:
    local_comments += f"_bs_cvae{size_batch_cvae:d}"

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
    strategy_primary=args.hindsight_strategy,
    additional_goals=additional_goals,
    num_goals_per_transition=1,
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
    activation=activation,
    num_classes_abstract=num_classes_abstract,
    argmax_reconstruction=argmax_reconstruction,
)

cvae.to(DEVICE)
params_cvae = cvae.parameters()
optimizer_cvae = torch.optim.Adam(params_cvae, lr=args.lr, eps=eps_adam)

################################################################
@torch.no_grad()
def visualize_classes_SSM_simple(maps_situations, env, num_categoricals, num_categories):
    def get_img_from_fig(fig, dpi=180):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def discrete_matshow(data, vmin, vmax):
        cmap = plt.get_cmap('RdBu', vmax - vmin + 1)
        plt.matshow(np.flip(data.T, 0), fignum=0, cmap=cmap, vmin=vmin - 0.5, vmax=vmax + 0.5)
        # plt.savefig("mask.png", bbox_inches='tight', pad_inches=0)
        rgb_array = get_img_from_fig(plt.gcf(), 100)
        plt.close()
        return rgb_array

    i_curr, j_curr, x_curr = env.obs2ijxd(env.obs_curr)

    overlays, masks = [], []

    for x in range(4):
        obs_base = env.ijxd2obs(i_curr, j_curr, x)
        background_rendered = env.render_image(ijs=None, obs=obs_base, no_agent=True)
        background_rendered[:, :2, :] = 255
        background_rendered[:, -2:, :] = 255
        background_rendered[:2, :, :] = 255
        background_rendered[-2:, :, :] = 255
        plt.axis("off")
        plt.margins(0)
    
        mask_rendered = discrete_matshow(maps_situations[:, :, x], vmin=0, vmax=num_categories ** num_categoricals - 1)
        mask_rendered = cv2.resize(mask_rendered, (background_rendered.shape[0], background_rendered.shape[1]), interpolation=cv2.INTER_NEAREST)
        mask_rendered[:, :2, :] = 255
        mask_rendered[:, -2:, :] = 255
        mask_rendered[:2, :, :] = 255
        mask_rendered[-2:, :, :] = 255
        combined = cv2.addWeighted(background_rendered, 0.5, mask_rendered, 0.5, 0)
        overlays.append(combined)
        masks.append(mask_rendered)
    overlays = np.concatenate(overlays, 1)
    masks = np.concatenate(masks, 1)
    return overlays, masks


milestones_evaluation = []
step_milestone, pointer_milestone = 0, 0
while step_milestone <= args.steps_stop:
    milestones_evaluation.append(step_milestone)
    step_milestone += args.freq_eval

path_writer = f"tb_records/{env.spec.id}/{args.size_world}x{args.size_world}_traindiff{args.difficulty:g}"
if args.num_envs_train > 0:
    path_writer += f"x{args.num_envs_train:d}"

path_writer += f"/{args.method}/vae_pretrain/{args.hindsight_strategy}/{args.seed}"
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
    obs_init = obs_curr
    if not (agent.steps_interact <= args.steps_max and episode_elapsed <= args.episodes_max and agent.steps_interact <= args.steps_stop):
        break
    while not done and agent.steps_interact <= args.steps_max:
        action = agent.decide(obs_curr, env=env, writer=writer, random_walk=args.random_walk)
        obs_next, reward, done, info = env.step(action)
        real_done = done and not info["overtime"]
        ################################################
        if agent.steps_interact - step_last_eval >= freq_visualize_generation and not real_done:
            idx_config = np.random.choice(range(len(configs_eval)))
            config_eval = configs_eval[idx_config]
            env_debug = func_get_new_env(args, **config_eval)
            obs_cond = env_debug.reset()
            # visualize_generation_minigrid2(cvae, obs_cond, env, writer, step_record=agent.steps_interact)
            step_last_eval += freq_visualize_generation
        sample = {"obs": obs_curr, "act": action, "rew": reward, "next_obs": obs_next, "done": real_done, "idx_env": env.idx_env}
        hrb.add(**sample)
        flag_debug = agent.steps_interact % 100 == 0
        if flag_debug:
            obs_curr_tensor = obs2tensor(obs_curr)
            codes_pred_tensor, obses_pred_tensor = cvae.generate_from_obs(obs_curr_tensor, num_samples=args.num_waypoints_unpruned)
            obses_pred_tensor = obses_pred_tensor.to(torch.uint8)
            obses_pred_tensor_unique = torch.unique(obses_pred_tensor.reshape(args.num_waypoints_unpruned, -1), dim=0)
            writer.add_scalar(f"Generation/num_wps_unpruned_unique", obses_pred_tensor_unique.shape[0], agent.steps_interact)
            writer.add_scalar(f"Generation/ratio_wps_unpruned_unique", obses_pred_tensor_unique.shape[0] / args.num_waypoints_unpruned, agent.steps_interact)
            mask_wps_unpruned_nonexistent = np.zeros(args.num_waypoints_unpruned, dtype=bool) # invalid for not reachable from obs_init
            states = env.obs2state(obses_pred_tensor.cpu().numpy())
            if env.name_game == "SwordShieldMonster":
                mask_wps_unpruned_irreversible = np.zeros(args.num_waypoints_unpruned, dtype=bool) # invalid for not reachable from obs_curr
                i_curr, j_curr, x_curr = env.obs2ijxd(obs_curr)
                i_pred, j_pred, x_pred, _ = env.state2ijxd(states)
            for idx_state in range(args.num_waypoints_unpruned):
                state_reachable_frominit = states[idx_state] in env.DP_info["states_reachable"]
                mask_wps_unpruned_nonexistent[idx_state] = not state_reachable_frominit
                if env.name_game == "SwordShieldMonster" and state_reachable_frominit: # efficient proxy for examining reachable target from now
                    x_targ = int(x_pred[idx_state])
                    targ_irreversible = False
                    if x_curr == 1 and (x_targ == 0 or x_targ == 2):
                        targ_irreversible = True
                    elif x_curr == 2 and (x_targ == 0 or x_targ == 1):
                        targ_irreversible = True
                    elif x_curr == 3 and x_targ < 3:
                        targ_irreversible = True
                    mask_wps_unpruned_irreversible[idx_state] = targ_irreversible
            num_wps_unpruned_nonexistent = int(mask_wps_unpruned_nonexistent.sum())
            writer.add_scalar(f"Generation/num_wps_unpruned_nonexistent", num_wps_unpruned_nonexistent, agent.steps_interact)
            writer.add_scalar(f"Generation/ratio_wps_unpruned_nonexistent", num_wps_unpruned_nonexistent / args.num_waypoints_unpruned, agent.steps_interact)
            if env.name_game == "SwordShieldMonster" and not mask_wps_unpruned_nonexistent.all():
                num_wps_unpruned_irreversible = int(mask_wps_unpruned_irreversible.sum())
                writer.add_scalar(f"Generation/num_wps_unpruned_irreversible", num_wps_unpruned_irreversible, agent.steps_interact)
                writer.add_scalar(f"Generation/ratio_wps_unpruned_irreversible", num_wps_unpruned_irreversible / args.num_waypoints_unpruned, agent.steps_interact)
                writer.add_scalar(f"Generation/ratio_wps_unpruned_irreversible_existent", num_wps_unpruned_irreversible / (args.num_waypoints_unpruned - num_wps_unpruned_nonexistent), agent.steps_interact)
        if agent.steps_interact >= agent.time_learning_starts and hrb.get_stored_size() > size_batch_cvae and agent.steps_interact % 4 == 0:
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
                    loss_entropy_weighted = (loss_entropy * weights_rb.detach().squeeze()).mean()
                    loss_recon_weighted = (loss_recon * weights_rb.detach().squeeze()).mean()
                else:
                    loss_entropy_weighted = loss_entropy.mean()
                    loss_recon_weighted = loss_recon.mean()
                if prioritized_cvae:
                    hrb.update_priorities(idxes_rb, loss_overall.detach().cpu().numpy().squeeze())

            writer.add_scalar(f"Loss/recon", loss_recon_weighted.item(), agent.steps_interact)
            writer.add_scalar(f"Loss/entropy", loss_entropy_weighted.item(), agent.steps_interact)
            writer.add_scalar(f"Loss/overall", loss_overall_weighted.item(), agent.steps_interact)
            if flag_debug:
                writer.add_scalar(f"Dist/L1", dist_L1_mean.item(), agent.steps_interact)
                writer.add_scalar(f"Dist/L1_nontrivial", dist_L1_nontrivial.item(), agent.steps_interact)
                writer.add_scalar(f"Dist/L1_trivial", dist_L1_trivial.item(), agent.steps_interact)
                writer.add_scalar(f"Generation/ratio_imperfect_recon", 1 - ratio_perfect_recon.item(), agent.steps_interact)
                writer.add_scalar(f"Generation/ratio_unaligned", 1 - ratio_aligned.item(), agent.steps_interact)
        ####################################
        steps_episode += 1
        agent.step(obs_curr, action, reward, obs_next, done and not info["overtime"], writer=writer)
        return_cum += reward
        return_cum_discount += reward * args.gamma ** env.step_count
        obs_curr = obs_next
    if done:
        agent.on_episode_end()
        hrb.on_episode_end()
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

        if args.method in ["random", "LEAP"]:
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
time_duration = time_end - time_start
print("total time elapsed: %s" % str(datetime.timedelta(seconds=time_duration)))
# NOTE(H): save the envs as well!
torch.save(
    {
        "steps_interact": agent.steps_interact,
        "model_state_dict": cvae.state_dict(),
        "num_categoricals": num_categoricals,
        "num_categories": num_categories,
    },
    os.path.join(path_writer, "cvae.pt"),
)
if args.num_envs_train > 0:
    with open(os.path.join(path_writer, "envs.pkl"), "wb") as file:
        pickle.dump(envs_train, file)
