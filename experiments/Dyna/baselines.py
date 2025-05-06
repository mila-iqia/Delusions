import torch, numpy as np, copy, os
import warnings
from utils import LinearSchedule, minigridobs2tensor, get_cpprb, RL_AGENT
from models import Encoder_MiniGrid, Binder_MiniGrid, Predictor_MiniGrid

class RW_AGENT(RL_AGENT):
    def __init__(self, env, gamma=0.99, seed=42, **kwargs):
        super(RW_AGENT, self).__init__(env, gamma, seed)
        self.steps_interact = 0
        self.time_learning_starts = 20000

    def decide(self, *args, **kwargs):
        return self.action_space.sample()

    def step(self, *args, **kwargs):
        self.steps_interact += 1


class DQN_BASE(RL_AGENT):
    def __init__(
        self,
        env,
        network_policy,
        gamma=0.99,
        clip_reward=False,
        exploration_fraction=0.02,
        epsilon_final_train=0.01,
        epsilon_eval=0.0,
        steps_total=50000000,
        size_buffer=1000000,
        prioritized_replay=True,
        func_obs2tensor=minigridobs2tensor,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=42,
    ):
        super(DQN_BASE, self).__init__(env, gamma, seed)

        self.clip_reward = clip_reward
        self.schedule_epsilon = LinearSchedule(
            schedule_timesteps=int(exploration_fraction * steps_total),
            initial_p=1.0,
            final_p=epsilon_final_train,
        )
        self.epsilon_eval = epsilon_eval
        self.device = device

        self.network_policy = network_policy.to(self.device)
        self.network_target = None

        self.steps_interact, self.steps_total = (
            0,
            steps_total,
        )

        self.step_last_print, self.time_last_print = 0, None

        self.obs2tensor = func_obs2tensor

        self.prioritized_replay = prioritized_replay
        self.rb = get_cpprb(env, size_buffer, prioritized=self.prioritized_replay)
        if self.prioritized_replay:
            self.size_batch_rb = 64
            self.batch_rb = get_cpprb(env, self.size_batch_rb, prioritized=False)
        if self.prioritized_replay:
            self.schedule_beta_sample_priorities = LinearSchedule(steps_total, initial_p=0.4, final_p=1.0)

    def save2disk(self, path):
        torch.save(self.network_policy.state_dict(), os.path.join(path, "policynet.pt"))
        
    def loadfromdisk(self, path):
        self.network_policy.load_state_dict(torch.load(os.path.join(path, "policynet.pt")))

    def add_to_buffer(self, batch, attach_priorities=True):
        if self.prioritized_replay and attach_priorities:
            self.batch_rb.add(**batch)
            if self.batch_rb.get_stored_size() >= self.size_batch_rb:
                batch = self.batch_rb.get_all_transitions()
                self.batch_rb.clear()
                (
                    batch_obs_curr,
                    batch_action,
                    batch_reward,
                    batch_obs_next,
                    batch_done,
                    weights,
                    batch_idxes,
                ) = self.process_batch(batch, prioritized=False)
                priorities = self.calculate_priorities(
                    batch_obs_curr,
                    batch_action,
                    batch_reward,
                    batch_obs_next,
                    batch_done,
                    error_absTD=None,
                )
                self.rb.add(**batch, priorities=priorities)
        else:
            self.rb.add(**batch)

    def on_episode_end(self, eval=False):
        self.rb.on_episode_end()

    def state_value(self, obs, done=None, network="double", clip=False):
        if network == "policy":
            network = self.network_policy
        elif network == "target":
            network = self.network_target
        elif network == "double":
            network = None
            network1 = self.network_target
            network2 = self.network_policy
        else:
            raise ValueError("what is this network?")
        if network is not None:
            predicted_Q = network(obs, scalarize=True)
            if clip:
                predicted_Q = torch.clamp(predicted_Q, network.value_min, network.value_max)
            with torch.no_grad():
                action_next = torch.argmax(predicted_Q.detach(), dim=1, keepdim=True)
            predicted_V = predicted_Q.gather(1, action_next)
        else:
            assert network1 is not None and network2 is not None
            predicted_Q2 = network2(obs, scalarize=True)
            if clip:
                predicted_Q2 = torch.clamp(predicted_Q2, network2.value_min, network2.value_max)
            with torch.no_grad():
                action_next = torch.argmax(predicted_Q2.detach(), dim=1, keepdim=True)
            predicted_Q1 = network1(obs, scalarize=True)
            predicted_V = predicted_Q1.gather(1, action_next)
            if clip:
                predicted_V = torch.clamp(predicted_V, network1.value_min, network1.value_max)
        if done is not None:
            assert done.shape[0] == obs.shape[0]
            predicted_V = torch.where(
                done.reshape(predicted_V.shape),
                torch.tensor(0.0, dtype=torch.float32, device=self.device),
                predicted_V,
            )
        return predicted_V

    def calculate_loss_TD(
        self,
        batch_obs_curr,
        batch_action,
        batch_reward,
        batch_obs_next,
        batch_done,
        type="kld",
        also_return_states=False,
    ):
        with torch.no_grad():
            values_next = self.state_value(batch_obs_next, batch_done, network="double", clip=True)
            target_TD = (batch_reward + self.gamma * values_next).detach()
        if type == "l1":
            values_curr, states = self.network_policy(batch_obs_curr, scalarize=True, also_return_states=True)
            values_curr = values_curr.gather(1, batch_action)
            loss_TD = torch.nn.functional.l1_loss(values_curr, target_TD, reduction="none")
        elif type == "kld":
            value_logits_curr, states = self.network_policy(batch_obs_curr, scalarize=False, also_return_states=True)
            value_logits_curr = value_logits_curr[torch.arange(batch_obs_curr.shape[0]), batch_action.squeeze()]
            with torch.no_grad():
                value_dist_target = self.network_policy.estimator_Q.histogram_converter.to_histogram(target_TD)
            loss_TD = torch.nn.functional.kl_div(torch.log_softmax(value_logits_curr, -1), value_dist_target.detach(), reduction="none").sum(-1, keepdims=True)
        elif type == "huber":
            values_curr, states = self.network_policy(batch_obs_curr, scalarize=True, also_return_states=True)
            values_curr = values_curr.gather(1, batch_action)
            loss_TD = torch.nn.functional.smooth_l1_loss(values_curr, target_TD, reduction="none")
        else:
            raise NotImplementedError("what is this loss type?")
        if also_return_states:
            return loss_TD, states
        else:
            return loss_TD
        
    @torch.no_grad()
    def calculate_TD_L1_scalar(self, batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done):
        values_next = self.state_value(batch_obs_next, batch_done, network="double", clip=True)
        target_TD = (batch_reward.squeeze() + self.gamma * values_next.squeeze())
        values_curr = self.network_policy(batch_obs_curr, action=batch_action, scalarize=True).squeeze()
        return torch.abs(target_TD - values_curr)

    @torch.no_grad()
    def calculate_priorities(
        self,
        batch_obs_curr,
        batch_action,
        batch_reward,
        batch_obs_next,
        batch_done,
        error_absTD=None,
    ):
        if error_absTD is None:
            error_absTD = self.calculate_TD_L1_scalar(batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done)
        else:
            assert error_absTD.shape[0] == batch_reward.shape[0]
        new_priorities = error_absTD.detach().cpu().numpy() + 1e-6
        return new_priorities

    @torch.no_grad()
    def process_batch(self, batch, prioritized=False):
        if prioritized:
            batch_obs_curr, batch_action, batch_reward, batch_done, batch_obs_next, weights, batch_idxes = batch.values()
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device).reshape(-1, 1)
        else:
            batch_obs_curr, batch_action, batch_reward, batch_done, batch_obs_next = batch.values()
            weights, batch_idxes = None, None
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32, device=self.device).reshape(-1, 1)
        batch_done = torch.tensor(batch_done, dtype=torch.bool, device=self.device).reshape(-1, 1)
        batch_action = torch.tensor(batch_action, dtype=torch.int64, device=self.device).reshape(-1, 1)
        batch_obs_curr, batch_obs_next = self.obs2tensor(batch_obs_curr, device=self.device), self.obs2tensor(batch_obs_next, device=self.device)
        if self.clip_reward:
            batch_reward = torch.sign(batch_reward)
        return batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, weights, batch_idxes

    def decide(self, obs, eval=False, env=None, writer=None, random_walk=False):
        if np.random.random() > float(eval) * self.epsilon_eval + (1 - float(eval)) * self.schedule_epsilon.value(self.steps_interact):
            with torch.no_grad():
                return int(torch.argmax(self.network_policy(self.obs2tensor(obs, device=self.device))))
        else:
            return self.action_space.sample()

    def step(self, obs_curr, action, reward, obs_next, done, eval=False, writer=None, idx_env=None):
        if obs_next is not None:
            sample = {
                "obs": np.array(obs_curr),
                "act": action,
                "rew": reward,
                "done": done,
                "next_obs": np.array(obs_next),
            }
            if idx_env is not None:
                sample["idx_env"] = idx_env
            self.add_to_buffer(sample)
        self.steps_interact += 1


class DQN(DQN_BASE):
    def __init__(
        self,
        env,
        network_policy,
        gamma=0.99,
        clip_reward=False,
        exploration_fraction=0.02,
        epsilon_final_train=0.01,
        epsilon_eval=0.0,
        steps_total=50000000,
        size_buffer=1000000,
        prioritized_replay=True,
        type_optimizer="Adam",
        lr=5e-4,
        eps=1.5e-4,
        time_learning_starts=20000,
        freq_targetsync=8000,
        freq_train=4,
        size_batch=32,
        func_obs2tensor=minigridobs2tensor,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=42,
    ):
        super(DQN, self).__init__(
            env,
            network_policy,
            gamma=gamma,
            clip_reward=clip_reward,
            exploration_fraction=exploration_fraction,
            epsilon_final_train=epsilon_final_train,
            epsilon_eval=epsilon_eval,
            steps_total=steps_total,
            size_buffer=size_buffer,
            prioritized_replay=prioritized_replay,
            func_obs2tensor=func_obs2tensor,
            device=device,
            seed=seed,
        )

        self.optimizer = eval("torch.optim.%s" % type_optimizer)(self.network_policy.parameters(), lr=lr, eps=eps)

        self.network_target = copy.deepcopy(self.network_policy)
        self.network_target.to(self.device)
        for param in self.network_target.parameters():
            param.requires_grad = False
        self.network_target.eval()
        for module in self.network_target.modules():
            module.eval()

        self.size_batch = size_batch
        self.time_learning_starts = time_learning_starts
        self.freq_train = freq_train
        self.freq_targetsync = freq_targetsync
        self.step_last_update = self.time_learning_starts - self.freq_train
        self.step_last_targetsync = self.time_learning_starts - self.freq_targetsync

    def step(self, obs_curr, action, reward, obs_next, done, eval=False, writer=None, idx_env=None):
        if obs_next is not None:
            sample = {
                "obs": np.array(obs_curr),
                "act": action,
                "rew": reward,
                "done": done,
                "next_obs": np.array(obs_next),
            }
            if idx_env is not None:
                sample["idx_env"] = idx_env
            self.add_to_buffer(sample)
        if self.steps_interact >= self.time_learning_starts:
            if self.rb.get_stored_size() >= self.size_batch and (self.steps_interact - self.step_last_update) >= self.freq_train:
                debug = np.random.random() < 0.05
                self.update(writer=writer, debug=debug)
                self.step_last_update += self.freq_train
            if (self.steps_interact - self.step_last_targetsync) >= self.freq_targetsync:
                self.sync_parameters()
                self.step_last_targetsync += self.freq_targetsync
        self.steps_interact += 1

    def sample_batch(self, size_batch=None):
        if size_batch is None:
            size_batch = self.size_batch
        if self.prioritized_replay:
            batch = self.rb.sample(
                size_batch,
                beta=self.schedule_beta_sample_priorities.value(self.steps_interact),
            )
        else:
            batch = self.rb.sample(self.size_batch)
        return batch

    def update(self, batch=None, writer=None, debug=False):
        if batch is None:
            batch = self.sample_batch()
        batch_processed = self.process_batch(batch, prioritized=self.prioritized_replay)
        batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, weights, batch_idxes = batch_processed
        

        type_TD_loss = "huber"
        dict_head = self.network_policy.estimator_Q.dict_head
        if dict_head["name"] == "Q" and dict_head["dist_out"]:
            type_TD_loss = "kld"

        error_TD = self.calculate_loss_TD(batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, type=type_TD_loss)

        if self.prioritized_replay:
            assert weights is not None
            error_TD_weighted = (error_TD * weights).mean()
        else:
            error_TD_weighted = error_TD.mean()

        self.optimizer.zero_grad(set_to_none=True)
        error_TD_weighted.backward()

        torch.nn.utils.clip_grad_value_(self.network_policy.parameters(), 1.0)
        self.optimizer.step()

        with torch.no_grad():
            if self.prioritized_replay:
                new_priorities = self.calculate_priorities(
                    batch_obs_curr,
                    batch_action,
                    batch_reward,
                    batch_obs_next,
                    batch_done,
                    error_absTD=None,
                )
                self.rb.update_priorities(batch_idxes, new_priorities.squeeze())

            if debug and writer is not None:
                writer.add_scalar(
                    "Loss/TD",
                    error_TD_weighted.item(),
                    self.step_last_update,
                )
                error_absTD = self.calculate_loss_TD(
                    batch_obs_curr,
                    batch_action,
                    batch_reward,
                    batch_obs_next,
                    batch_done,
                    type="l1",
                )
                writer.add_scalar(
                    "Debug/res_TD",
                    error_absTD.mean().item(),
                    self.step_last_update,
                )

    def sync_parameters(self):
        self.network_target.load_state_dict(self.network_policy.state_dict())
        for param in self.network_target.parameters():
            param.requires_grad = False
        self.network_target.eval()
        print("policy-target parameters synced")


class DQN_NETWORK(torch.nn.Module):
    def __init__(self, encoder, estimator_Q, binder=None):
        super(DQN_NETWORK, self).__init__()
        self.encoder, self.estimator_Q = encoder, estimator_Q
        self.binder = binder
        self.value_min, self.value_max = self.estimator_Q.value_min, self.estimator_Q.value_max

    def forward(self, obs, action=None, scalarize=True, only_value=True, also_return_states=False):
        state = self.encoder(obs)
        if self.binder is None:
            state_local = state
        else:
            state_local = self.binder.extract_local_field(state)
        ret = self.estimator_Q(state_local, action=action, scalarize=scalarize)
        if only_value and isinstance(ret, tuple):
            ret = ret[0]
        if also_return_states:
            if isinstance(ret, tuple):
                return *ret, state_local
            else:
                return ret, state_local
        else:
            return ret

    def parameters(self):
        parameters = []
        parameters += list(self.encoder.parameters())
        if self.binder is not None:
            parameters += list(self.binder.parameters())
        parameters += list(self.estimator_Q.parameters())
        return parameters


def create_RW_agent(args, env, **kwargs):
    return RW_AGENT(env, args.gamma, args.seed)


def create_DQN_agent(args, env, dim_embed, num_actions, device=None):
    if device is None:
        if torch.cuda.is_available() and not args.force_cpu:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            warnings.warn("agent created on cpu")

    if args.activation == "relu":
        activation = torch.nn.ReLU
    elif args.activation == "elu":
        activation = torch.nn.ELU
    elif args.activation == "leakyrelu":
        activation = torch.nn.LeakyReLU
    elif args.activation == "silu":
        activation = torch.nn.SiLU

    encoder = Encoder_MiniGrid(
        dim_embed,
        sample_obs=env.reset(),
        norm=bool(args.layernorm),
        append_pos=False,
        activation=activation,
    )
    encoder.to(device)

    sample_input = encoder(minigridobs2tensor(env.obs_curr))

    binder = Binder_MiniGrid(
        sample_input,
        len_rep=args.len_rep,
        norm=bool(args.layernorm),
        activation=activation,
        num_heads=args.num_heads,
        size_bottleneck=args.size_bottleneck,
        type_arch=args.arch_enc,
    )
    binder.to(device)

    dict_head_Q = {
        "name": "Q",
        "len_predict": num_actions,
        "dist_out": True,
        "value_min": args.value_min,
        "value_max": args.value_max,
        "atoms": args.atoms_value,
        "classify": False,
    }
    estimator_Q = Predictor_MiniGrid(
        num_actions,
        len_input=int(binder.len_out // 2),
        depth=args.depth_hidden,
        width=args.width_hidden,
        norm=bool(args.layernorm),
        activation=activation,
        dict_head=dict_head_Q,
        value_min=args.value_min,
        value_max=args.value_max,
    )
    estimator_Q.to(device)

    agent = DQN(
        env,
        DQN_NETWORK(encoder, estimator_Q, binder=binder),
        gamma=args.gamma,
        steps_total=args.steps_max,
        prioritized_replay=bool(args.prioritized_replay),
        lr=args.lr,
        size_batch=args.size_batch,
        device=device,
        seed=args.seed,
    )
    return agent
