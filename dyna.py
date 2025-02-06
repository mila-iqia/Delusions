import torch, numpy as np, copy, os
import warnings
from utils import LinearSchedule, minigridobs2tensor, RL_AGENT, process_batch

class MINIGRID_OBS_ONESTEP_MODEL(torch.nn.Module):
    def __init__(
        self,
        layout_extractor,
        decoder,
        sample_input,
        num_actions=4,
        len_action=16,
        dim_embed_bow=16,
        activation=torch.nn.ReLU,
        argmax_reconstruction=False,
        num_classes_abstract=4,
        prob_random_generation=0.5,
        len_rep=128,
        **kwargs,
    ):
        super(MINIGRID_OBS_ONESTEP_MODEL, self).__init__(**kwargs) # TODO(H): have to predict rewards and termination as well

        from models import Encoder_MiniGrid, Predictor_MiniGrid, Embedder_MiniGrid_BOW

        self.argmax_reconstruction = argmax_reconstruction
        self.dim_embed_bow = dim_embed_bow
        self.len_rep = len_rep

        self.num_actions = num_actions
        self.len_action = len_action
        self.embedding_actions = torch.nn.Embedding(self.num_actions, self.len_action)

        self.num_classes_abstract = num_classes_abstract
        self.decoder = decoder
        self.layout_extractor = layout_extractor
        self.prob_random_generation = prob_random_generation

        from minigrid import OBJECT_TO_IDX

        self.object_to_idx = OBJECT_TO_IDX
        self.steps_trained = 0
        self.size_input = sample_input.shape[-2]

        self.encoder_context = Embedder_MiniGrid_BOW(
            dim_embed=self.dim_embed_bow, width=sample_input.shape[-3], height=sample_input.shape[-2], channels_obs=sample_input.shape[-1], ebd_pos=False
        )

        self.decompressor = torch.nn.Sequential(
            torch.nn.Unflatten(1, (self.len_action, 1, 1)),
            torch.nn.ConvTranspose2d(self.len_action, self.dim_embed_bow, kernel_size=self.size_input, stride=1, padding=0),
        )

        self.fuser = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32 * self.size_input ** 2, self.num_classes_abstract * self.size_input ** 2),
        )

        self.encoder_state = torch.nn.Sequential(
            Encoder_MiniGrid(self.dim_embed_bow, sample_input, norm=True, append_pos=False, activation=activation),
            activation(False),
            torch.nn.Flatten(),
            torch.nn.Linear(self.dim_embed_bow * sample_input.shape[-2] * sample_input.shape[-3], self.len_rep),
        )


        self.predictor_reward = Predictor_MiniGrid(1, self.len_rep, depth=3, width=256, activation=activation, norm=True,
        dict_head={"len_predict": None, "dist_out": True, "value_min": 0.0, "value_max": 1.0, "atoms": 2, "classify": False},
        value_min=0.0, value_max=1.0) # TODO(H): simplest would be just to use the obs_pred
        self.predictor_omega = Predictor_MiniGrid(1, self.len_rep, depth=3, width=256, activation=activation, norm=True,
        dict_head={"len_predict": None, "dist_out": True, "value_min": 0.0, "value_max": 1.0, "atoms": 2, "classify": False},
        value_min=0.0, value_max=1.0)


    def to(self, device):
        super().to(device)
        self.encoder_context.to(device)
        self.decompressor.to(device)
        self.fuser.to(device)
        self.encoder_state.to(device)
        self.predictor_reward.to(device)
        self.predictor_omega.to(device)

    def parameters(self):
        parameters = []
        parameters += list(self.encoder_context.parameters())
        parameters += list(self.decompressor.parameters())
        parameters += list(self.fuser.parameters())
        parameters += list(self.encoder_state.parameters())
        parameters += list(self.predictor_reward.parameters())
        parameters += list(self.predictor_omega.parameters())
        return parameters

    # @profile
    def logits_from_action(self, action, context):
        size_batch = action.shape[0]
        assert context.shape[0] == size_batch
        ebd_action = self.embedding_actions(action).reshape(size_batch, -1)
        samples_decompressed = self.decompressor(ebd_action) # TODO(H): the effectiveness of the architecture is to be tested with experiments
        logits_mask_pos_agent = self.fuser(torch.cat([samples_decompressed, context], 1))
        return logits_mask_pos_agent

    @torch.no_grad()
    # @profile
    def mask_from_logits(self, logits_mask_pos_agent, noargmax=False):
        size_batch = logits_mask_pos_agent.shape[0]
        logits_mask_pos_agent = logits_mask_pos_agent.reshape(size_batch, -1)
        assert logits_mask_pos_agent.shape[-1] == self.num_classes_abstract * self.size_input ** 2
        if np.random.rand() < self.prob_random_generation:
            probs_mask_pos_agent = torch.ones_like(logits_mask_pos_agent) / torch.numel(logits_mask_pos_agent)
            mask_agent_pred = torch.distributions.OneHotCategorical(probs=probs_mask_pos_agent).sample()
        else:
            if self.argmax_reconstruction and not noargmax:
                mask_agent_pred = torch.nn.functional.one_hot(logits_mask_pos_agent.argmax(-1), num_classes=self.num_classes_abstract * self.size_input ** 2)
            else:
                mask_agent_pred = torch.distributions.OneHotCategorical(logits=logits_mask_pos_agent).sample()
        mask_agent_pred = mask_agent_pred.reshape(-1, self.size_input, self.size_input, self.num_classes_abstract)
        return mask_agent_pred

    @torch.no_grad()
    def fuse_mask_with_obs(self, mask, obs):
        layout, _ = self.layout_extractor(obs)
        return self.decoder(layout, mask)

    # @profile
    def forward(self, obs_curr, action, obs_next=None, train=False):
        size_batch = obs_curr.shape[0]
        context = self.encoder_context(obs_curr)
        action = action.reshape(size_batch, -1)
        logits_mask_pos_agent = self.logits_from_action(action, context).reshape(size_batch, -1)
        
        layouts_curr, _ = self.layout_extractor(obs_curr)
        if train:
            assert obs_next is not None
            state_next = self.encoder_state(obs_next)
            logits_reward = self.predictor_reward(state_next, scalarize=False)
            logits_omega = self.predictor_omega(state_next, scalarize=False)
            return logits_mask_pos_agent, logits_reward, logits_omega
        else:
            assert obs_next is None
            with torch.no_grad():
                mask_agent_pred = self.mask_from_logits(logits_mask_pos_agent)
            obs_next = self.decoder(layouts_curr, mask_agent_pred)
            state_next = self.encoder_state(obs_next)
            reward = self.predictor_reward(state_next, scalarize=True)
            term = self.predictor_omega(state_next, scalarize=True)
            return mask_agent_pred.bool(), reward, term

    def imagine_transition_from_obs(self, obses): # TODO(H): not validated
        size_batch = obses.shape[0]
        action_sampled = torch.randint(0, self.num_actions, (size_batch, 1), device=obses.device) # shape is good for return
        context = self.encoder_context(obses)
        logits_mask_pos_agent = self.logits_from_action(action_sampled, context).reshape(size_batch, -1)
        with torch.no_grad():
            mask_agent_pred = self.mask_from_logits(logits_mask_pos_agent)
        obses_pred = self.fuse_mask_with_obs(mask_agent_pred.bool(), obses)
        state_pred = self.encoder_state(obses_pred)
        reward_pred = self.predictor_reward(state_pred, scalarize=True)
        omega_pred = self.predictor_omega(state_pred, scalarize=False).argmax(-1).reshape(-1).bool()
        return action_sampled, reward_pred, obses_pred, omega_pred

    # @profile
    def compute_loss(self, batch_processed, debug=False):
        batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, batch_obs_targ, batch_obs_targ2, weights, batch_idxes = batch_processed
        with torch.no_grad():
            obses_context, obses_target = batch_obs_curr, batch_obs_next
        size_batch = obses_target.shape[0]
        layouts_train, masks_agent_target = self.layout_extractor(obses_target)
        logits_mask_pos_agent_train, logits_reward, logits_omega = self.forward(obs_curr=obses_context, action=batch_action, obs_next=batch_obs_next, train=True)
        logsoftmax_mask_agent = logits_mask_pos_agent_train.log_softmax(-1)
        loss_recon = torch.nn.functional.kl_div(
            input=logsoftmax_mask_agent, target=masks_agent_target.float().reshape(size_batch, -1), log_target=False, reduction="none"
        ).sum(-1)
        with torch.no_grad():
            target_reward = self.predictor_reward.histogram_converter.to_histogram(batch_reward)
        logits_reward = logits_reward.reshape(size_batch, -1)
        loss_reward = torch.nn.functional.kl_div(torch.log_softmax(logits_reward, -1), target_reward.detach(), reduction="none").sum(-1)
        logits_omega = logits_omega.reshape(-1, 2)
        loss_omega = torch.nn.functional.cross_entropy(torch.log_softmax(logits_omega, -1), batch_done.to(torch.long).detach(), reduction="none")
        loss_overall = loss_recon + loss_reward + loss_omega

        self.steps_trained += 1

        if not debug:
            return loss_overall, loss_recon, loss_reward, loss_omega, None, None, None, None, None
        else:
            with torch.no_grad():
                masks_agent_pred = (
                    torch.nn.functional.one_hot(logits_mask_pos_agent_train.argmax(-1), obses_target.shape[1] * obses_target.shape[2] * self.num_classes_abstract)
                    .bool()
                    .reshape(size_batch, obses_target.shape[-3], obses_target.shape[-2], self.num_classes_abstract)
                )
                obses_pred = self.decoder(layouts_train, masks_agent_pred)
                omegas_pred = torch.argmax(logits_omega, dim=-1).bool()
                batch_not_done = ~batch_done
                diff_omega = (omegas_pred != batch_done)
                ratio_diff_omega = diff_omega.float().mean()
                if batch_not_done.any():
                    ratio_fp_omega = ((omegas_pred != batch_done)[batch_not_done]).sum() / batch_not_done.sum()
                else:
                    ratio_fp_omega = None
                if batch_done.any():
                    ratio_fn_omega = ((omegas_pred != batch_done)[batch_done]).sum() / (batch_done).sum()
                else:
                    ratio_fn_omega = None
                rewards_pred = logits_reward.softmax(-1) @ self.predictor_reward.histogram_converter.support
                diff_rewards = torch.abs(rewards_pred - batch_reward.reshape(-1)).mean()
                dist_L1 = torch.abs(obses_pred.float() - obses_target.float())
                mask_perfect_recon = dist_L1.reshape(dist_L1.shape[0], -1).sum(-1) == 0
                ratio_perfect_recon = mask_perfect_recon.sum() / mask_perfect_recon.shape[0]

            return loss_overall, loss_recon, loss_reward, loss_omega, ratio_perfect_recon, diff_rewards, ratio_diff_omega, ratio_fp_omega, ratio_fn_omega

class DQN_DYNA_NETWORK(torch.nn.Module):
    def __init__(self, encoder_Q, estimator_Q, encoder_evaluator=None, binder_evaluator=None, evaluator_discount=None, model=None):
        super(DQN_DYNA_NETWORK, self).__init__()
        self.encoder_Q = encoder_Q
        self.estimator_Q = estimator_Q # NOTE(H): let estimator Q be obs2values, since it does not share encoder with the evaluator
        self.encoder_evaluator = encoder_evaluator
        self.binder_evaluator = binder_evaluator
        self.evaluator_discount = evaluator_discount
        self.model = model

    def to(self, device):
        super().to(device)
        self.encoder_Q.to(device)
        self.estimator_Q.to(device)
        if self.encoder_evaluator is not None:
            self.encoder_evaluator.to(device)
        if self.binder_evaluator is not None:
            self.binder_evaluator.to(device)
        if self.evaluator_discount is not None:
            self.evaluator_discount.to(device)
        if self.model is not None:
            self.model.to(device)

    def parameters(self):
        parameters = list(self.parameters_dyna()) + list(self.parameters_evaluator())
        return parameters
    
    def parameters_dyna(self):
        parameters = []
        parameters += list(self.encoder_Q.parameters())
        parameters += list(self.estimator_Q.parameters())
        if self.model is not None:
            parameters += list(self.model.parameters())
        return parameters
    
    def parameters_evaluator(self):
        parameters = []
        if self.encoder_evaluator is not None:
            parameters += list(self.encoder_evaluator.parameters())
        if self.binder_evaluator is not None:
            parameters += list(self.binder_evaluator.parameters())
        if self.evaluator_discount is not None:
            parameters += list(self.evaluator_discount.parameters())
        return parameters
    
    def network_Q(self, obs, action=None, scalarize=False):
        state = self.encoder_Q(obs)
        return self.estimator_Q(state, action, scalarize=scalarize)
    
    def network_evaluator(self, obs_curr, obs_targ, scalarize=False): 
        state_curr = self.encoder_evaluator(obs_curr)
        state_targ = self.encoder_evaluator(obs_targ)
        state_local_binded = self.binder_evaluator(state_curr, state_targ)
        return self.evaluator_discount(state_local_binded, scalarize=scalarize)

class DQN_DYNA_BASE(RL_AGENT):
    def __init__(
        self,
        env,
        network_policy,
        gamma=0.99,
        clip_reward=True,
        exploration_fraction=0.02,
        epsilon_final_train=0.01,
        epsilon_eval=0.001,
        steps_total=50000000,
        prioritized_replay=True,
        func_obs2tensor=minigridobs2tensor,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=42,
        hrb=None,
        silent=False,
        transform_discount_target=True,
        prob_relabel_generateJIT=0.0,
    ):
        super(DQN_DYNA_BASE, self).__init__(env, gamma, seed)

        self.clip_reward = clip_reward
        self.schedule_epsilon = LinearSchedule(
            schedule_timesteps=int(exploration_fraction * steps_total),
            initial_p=1.0,
            final_p=epsilon_final_train,
        )
        self.epsilon_eval = epsilon_eval

        self.device = device

        self.network_policy = network_policy
        self.network_target = self.network_policy

        if self.network_policy.evaluator_discount is not None:
            self.support_discount = self.network_policy.evaluator_discount.histogram_converter.support_discount
            self.support_distance = self.network_policy.evaluator_discount.histogram_converter.support_distance
        
        self.model = self.network_policy.model

        self.prob_relabel_generateJIT = float(prob_relabel_generateJIT)
        self.transform_discount_target = bool(transform_discount_target)

        self.steps_interact, self.steps_total = 0, steps_total  # steps_interact denotes the number of agent-env interactions
        self.steps_processed = 0

        self.step_last_print, self.time_last_print = 0, None

        self.obs2tensor = lambda obs: func_obs2tensor(obs, device=self.device)

        self.prioritized_replay = prioritized_replay
        self.hrb = hrb
        if self.prioritized_replay:
            self.schedule_beta_sample_priorities = LinearSchedule(steps_total, initial_p=0.4, final_p=1.0)
        self.silent = silent

        self.waypoints_existing, self.proxy_graph_curr = None, None

        self.on_episode_end(eval=True)  # NOTE: do not call hrb.on_episode_end() here when there is no experience

    def save2disk(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.network_policy.state_dict(), os.path.join(path, "policynet.pt"))

    def add_to_buffer(self, batch):
        self.hrb.add(**batch)

    @torch.no_grad()
    def process_batch(self, batch, prioritized=False, with_targ=False):
        return process_batch(
            batch, prioritized=prioritized, with_targ=with_targ, device=self.device, obs2tensor=minigridobs2tensor, clip_reward=self.clip_reward)

    @torch.no_grad()
    def on_episode_end(self, eval=False):
        if self.hrb is not None and not eval:
            self.hrb.on_episode_end()

    def state_value(self, obs, done=None, network="double", clip=False):
        if network == "policy":
            network = self.network_policy
        elif network == "target":
            network = self.network_target
        elif network == "double": # Double DQN
            network = None
            network1 = self.network_target
            network2 = self.network_policy
        else:
            raise ValueError("what is this network?")
        if network is not None:
            predicted_Q = network.network_Q(obs, scalarize=True)
            if clip:
                predicted_Q = torch.clamp(predicted_Q, network.estimator_Q.value_min, network.estimator_Q.value_max)
            with torch.no_grad():
                action_next = torch.argmax(predicted_Q.detach(), dim=1, keepdim=True)
            predicted_V = predicted_Q.gather(1, action_next)
        else:
            assert network1 is not None and network2 is not None
            predicted_Q2 = network2.network_Q(obs, scalarize=True)
            if clip:
                predicted_Q2 = torch.clamp(predicted_Q2, network2.estimator_Q.value_min, network2.estimator_Q.value_max)
            with torch.no_grad():
                action_next = torch.argmax(predicted_Q2.detach(), dim=1, keepdim=True)
            predicted_Q1 = network1.network_Q(obs, scalarize=True)
            predicted_V = predicted_Q1.gather(1, action_next)
            if clip:
                predicted_V = torch.clamp(predicted_V, network1.estimator_Q.value_min, network1.estimator_Q.value_max)
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
    ):
        with torch.no_grad():
            values_next = self.state_value(batch_obs_next, batch_done, network="double", clip=True)
            target_TD = (batch_reward + self.gamma * values_next).detach()
        if type == "l1":
            values_curr = self.network_policy.network_Q(batch_obs_curr, scalarize=True)
            values_curr = values_curr.gather(1, batch_action)
            loss_TD = torch.nn.functional.l1_loss(values_curr, target_TD, reduction="none")
        elif type == "kld":
            value_logits_curr = self.network_policy.network_Q(batch_obs_curr, scalarize=False)
            value_logits_curr = value_logits_curr[torch.arange(batch_obs_curr.shape[0]), batch_action.squeeze()]
            with torch.no_grad():
                value_dist_target = self.network_policy.estimator_Q.histogram_converter.to_histogram(target_TD)
            loss_TD = torch.nn.functional.kl_div(torch.log_softmax(value_logits_curr, -1), value_dist_target.detach(), reduction="none").sum(-1, keepdims=True)
        elif type == "huber":
            values_curr = self.network_policy.network_Q(batch_obs_curr, scalarize=True)
            values_curr = values_curr.gather(1, batch_action)
            loss_TD = torch.nn.functional.smooth_l1_loss(values_curr, target_TD, reduction="none")
        else:
            raise NotImplementedError("what is this loss type?")
        return loss_TD

    @torch.no_grad()
    def calculate_TD_L1_scalar(self, batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done):
        values_next = self.state_value(batch_obs_next, batch_done, network="double", clip=True)
        target_TD = (batch_reward.squeeze() + self.gamma * values_next.squeeze())
        values_curr = self.network_policy.network_Q(batch_obs_curr, action=batch_action, scalarize=True).squeeze()
        return torch.abs(target_TD - values_curr)

    # @profile
    def calculate_multihead_error(
        self,
        batch_obs_curr,
        batch_action,
        batch_reward,
        batch_obs_next,
        batch_done,
        batch_obs_targ,
        freeze_encoder=False,
        freeze_binder=False,
        state_local_curr=None,
        state_local_next=None,
        state_local_next_targetnet=None,
    ):
        size_batch = batch_obs_curr.shape[0]

        if state_local_curr is not None or state_local_next is not None or state_local_next_targetnet is not None:
            assert state_local_curr is not None and state_local_next is not None and state_local_next_targetnet is not None # pass all 3
            batch_state_curr = None
            flag_reuse = True
        else:
            flag_reuse = False

        with torch.no_grad():
            batch_targ_reached = (batch_obs_next == batch_obs_targ).reshape(size_batch, -1).all(-1)
            if not flag_reuse:
                batch_obs_next_targ = torch.cat([batch_obs_next, batch_obs_targ], 0)
                batch_obs_curr_next_targ = torch.cat([batch_obs_curr, batch_obs_next_targ], 0)

        with torch.set_grad_enabled(not freeze_encoder):
            if flag_reuse:
                batch_state_targ = self.network_policy.encoder_evaluator(batch_obs_targ)
            else:
                batch_state_curr_next_targ = self.network_policy.encoder_evaluator(batch_obs_curr_next_targ)
                batch_state_curr, batch_state_next, batch_state_targ = batch_state_curr_next_targ.chunk(3, dim=0)

        with torch.set_grad_enabled(not freeze_binder):
            if flag_reuse:
                if self.network_policy.binder_evaluator.local_perception:
                    state_local_targ = self.network_policy.binder_evaluator.extract_local_field(batch_state_targ)
                else:
                    state_local_targ = self.network_policy.binder_evaluator.flattener(batch_state_targ)
                states_local_curr_targ = torch.cat([state_local_curr, state_local_targ], -1)
            else:
                if self.network_policy.binder_evaluator.local_perception:
                    state_local_curr_next_targ = self.network_policy.binder_evaluator.extract_local_field(batch_state_curr_next_targ)
                else:
                    state_local_curr_next_targ = self.network_policy.binder_evaluator.flattener(batch_state_curr_next_targ)
                state_local_curr, state_local_next, state_local_targ = torch.split(state_local_curr_next_targ, [size_batch, size_batch, size_batch], dim=0)
                states_local_curr_targ = torch.cat([state_local_curr, state_local_targ], -1)

        predicted_discount = self.network_policy.evaluator_discount(states_local_curr_targ, batch_action, scalarize=False)

        with torch.no_grad():
            states_local_next_targ = torch.cat([state_local_next.detach(), state_local_targ.detach()], -1)
            softmax_predicted_discount_next = self.network_policy.evaluator_discount(states_local_next_targ.detach(), scalarize=False).softmax(-1)
            predicted_discount_next = softmax_predicted_discount_next @ self.support_discount
            action_next = torch.argmax(predicted_discount_next.detach(), dim=1, keepdim=True)
            if flag_reuse:
                batch_state_targ_targetnet = self.network_target.encoder_evaluator(batch_obs_targ)
                if self.network_target.binder_evaluator.local_perception:
                    state_local_targ_targetnet = self.network_policy.binder_evaluator.extract_local_field(batch_state_targ_targetnet)
                else:
                    state_local_targ_targetnet = self.network_policy.binder_evaluator.flattener(batch_state_targ_targetnet)
            else:
                batch_state_next_targ_targetnet = self.network_target.encoder_evaluator(batch_obs_next_targ)
                if self.network_target.binder_evaluator.local_perception:
                    state_local_next_targ_targetnet = self.network_policy.binder_evaluator.extract_local_field(batch_state_next_targ_targetnet)
                else:
                    state_local_next_targ_targetnet = self.network_policy.binder_evaluator.flattener(batch_state_next_targ_targetnet)
                state_local_next_targetnet, state_local_targ_targetnet = torch.split(state_local_next_targ_targetnet, [size_batch, size_batch], dim=0)
            states_local_next_targ_targetnet = torch.cat([state_local_next_targetnet, state_local_targ_targetnet], -1)

        # discount head
        with torch.no_grad():
            dist_discounts = self.network_target.evaluator_discount(states_local_next_targ_targetnet, action_next, scalarize=False).softmax(-1)
            if self.transform_discount_target:
                distance_next = (dist_discounts @ self.support_distance).reshape(size_batch, 1)
                distance_next[batch_done] = 1000.0
                distance_next[batch_targ_reached] = 0.0
                target_discount_distance = 1.0 + distance_next
            else:
                discount_next = (dist_discounts @ self.network_target.evaluator_discount.histogram_converter.support_discount).reshape(size_batch, 1)
                discount_next[batch_done] = 0.0
                discount_next[batch_targ_reached] = 1.0
                target_discount_distance = self.gamma * discount_next
            target_discount_dist = self.network_target.evaluator_discount.histogram_converter.to_histogram(target_discount_distance)
        discount_logits_curr = predicted_discount.reshape(size_batch, -1)
        loss_discount = torch.nn.functional.kl_div(torch.log_softmax(discount_logits_curr, -1), target_discount_dist.detach(), reduction="none").sum(-1)
        return loss_discount, batch_state_curr, state_local_curr, state_local_next, state_local_next_targetnet


    @torch.no_grad()
    def decide(self, obs, eval=False, env=None, writer=None, random_walk=False):
        """
        input observation and output action
        some through the computations of the policy network
        """
        if np.random.random() > float(eval) * self.epsilon_eval + (1 - float(eval)) * self.schedule_epsilon.value(self.steps_interact):
            return int(torch.argmax(self.network_policy.network_Q(self.obs2tensor(obs), scalarize=True)))
        else:  # explore
            return self.action_space.sample()

    def step(self, obs_curr, action, reward, obs_next, done, writer=None, add_to_buffer=True, increment_steps=True, idx_env=None):
        if increment_steps:
            self.steps_interact += 1
        if add_to_buffer and obs_next is not None:
            sample = {"obs": np.array(obs_curr), "act": action, "rew": reward, "done": done, "next_obs": np.array(obs_next)}
            if idx_env is not None:
                sample["idx_env"] = idx_env
            self.add_to_buffer(sample) # TODO(H): can add the online computation of priorities here

class DQN_DYNA(DQN_DYNA_BASE):
    def __init__(
        self,
        env,
        network_policy,
        network_target=None,
        gamma=0.99,
        clip_reward=True,
        exploration_fraction=0.02,
        epsilon_final_train=0.01,
        epsilon_eval=0.001,
        steps_total=50000000,
        prioritized_replay=True,
        type_optimizer="Adam",
        lr=5e-4,
        eps=1.5e-4,
        time_learning_starts=20000,
        freq_targetsync=8000,
        freq_train=4,
        size_batch=64,
        func_obs2tensor=minigridobs2tensor,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=42,
        hrb=None,
        silent=False,
        transform_discount_target=True,
        prob_relabel_generateJIT=0.0,
    ):
        super(DQN_DYNA, self).__init__(
            env,
            network_policy,
            gamma=gamma,
            clip_reward=clip_reward,
            exploration_fraction=exploration_fraction,
            epsilon_final_train=epsilon_final_train,
            epsilon_eval=epsilon_eval,
            steps_total=steps_total,
            prioritized_replay=prioritized_replay,
            func_obs2tensor=func_obs2tensor,
            device=device,
            seed=seed,
            hrb=hrb,
            silent=silent,
            transform_discount_target=transform_discount_target,
            prob_relabel_generateJIT=prob_relabel_generateJIT,
        )

        self.parameters_dyna = self.network_policy.parameters_dyna()
        self.optimizer_dyna = eval("torch.optim.%s" % type_optimizer)(self.parameters_dyna, lr=lr, eps=eps)

        if self.network_policy.encoder_evaluator is not None and self.network_policy.binder_evaluator is not None and self.network_policy.evaluator_discount is not None:
            self.parameters_evaluator = self.network_policy.parameters_evaluator()
            self.optimizer_evaluator = eval("torch.optim.%s" % type_optimizer)(self.parameters_evaluator, lr=lr, eps=eps)
        else:
            self.parameters_evaluator, self.optimizer_evaluator = [], None

        # initialize target network
        if network_target is None:
            self.network_target = copy.deepcopy(self.network_policy)
        else:
            self.network_target = network_target
        # self.network_target.to(self.device)
        if self.network_target.model is not None:
            self.network_target.model.to("cpu")
            self.network_target.model = None
        for param in self.network_target.parameters():
            param.requires_grad = False
        self.network_target.eval()
        for module in self.network_target.modules():
            module.eval()

        self.size_batch = size_batch
        self.time_learning_starts = time_learning_starts
        assert self.time_learning_starts >= self.size_batch
        self.freq_train = freq_train
        self.freq_targetsync = freq_targetsync
        self.steps_processed = 0
        self.step_last_targetsync = self.time_learning_starts

    def need_update(self):
        if self.steps_interact >= self.time_learning_starts:
            if self.hrb.get_stored_size() >= self.size_batch and (self.steps_interact - self.steps_processed) >= self.freq_train:
                return True
        return False

    def update_step(self, batch_processed=None, writer=None):
        if self.steps_interact >= self.time_learning_starts:
            if self.steps_interact - self.step_last_targetsync >= self.freq_targetsync:
                self.sync_parameters()
                self.step_last_targetsync += self.freq_targetsync
            if self.steps_interact - self.steps_processed >= self.freq_train:
                self.update(batch_processed=batch_processed, writer=writer)
                if self.steps_processed == 0:
                    self.steps_processed = self.time_learning_starts
                else:
                    self.steps_processed += self.freq_train

    def step(self, obs_curr, action, reward, obs_next, done, writer=None, add_to_buffer=True, increment_steps=True, idx_env=None):
        """
        an agent step: in this step the agent does whatever it needs
        """
        super().step(obs_curr, action, reward, obs_next, done, writer=writer, add_to_buffer=add_to_buffer, increment_steps=increment_steps, idx_env=idx_env)
        self.update_step(writer=writer)

    # @profile
    def update(self, batch_processed=None, writer=None):
        """
        update the parameters of the DQN model using the weighted sampled Bellman error
        """
        debug = writer is not None and np.random.rand() < 0.01
        with torch.no_grad():
            if batch_processed is None:
                if self.prioritized_replay:
                    batch = self.hrb.sample(self.size_batch, beta=self.schedule_beta_sample_priorities.value(self.steps_interact))
                else:
                    batch = self.hrb.sample(self.size_batch)
                batch_processed = self.process_batch(batch, prioritized=self.prioritized_replay, with_targ=True)
            batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, batch_obs_targ, batch_obs_targ2, weights, batch_idxes = batch_processed

        ###### STEP 0: generate a batch imagined transitions, to be used by "generate_JIT" (according to prob) or Dyna (for sure)
        with torch.no_grad():
            assert self.model is not None
            action_imagined, reward_imagined, obses_imagined, omegas_imagined = self.model.imagine_transition_from_obs(batch_obs_curr) # TODO(H): check if shape is convenient

        ###### STEP 1: train evaluator (optional)
        if self.optimizer_evaluator is not None:
            if self.prob_relabel_generateJIT > 0:
                if np.random.rand() < float(self.prob_relabel_generateJIT):
                    batch_obs_targ2 = obses_imagined

            loss_discount, batch_state_curr, state_local_curr, _, _ = self.calculate_multihead_error(batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, batch_obs_targ2)
            
            if self.prioritized_replay:
                assert weights is not None
                error_evaluator_weighted = (loss_discount * weights.squeeze().detach()).mean() # kaixhin's rainbow implementation used mean()
            else:
                error_evaluator_weighted = loss_discount.mean()

            self.optimizer_evaluator.zero_grad(set_to_none=True)
            error_evaluator_weighted.backward()
            if debug:
                with torch.no_grad():
                    grads = [param.grad.detach().flatten() for param in self.parameters_evaluator] # if param.grad is not None
                    norm_grad_evaluator = torch.cat(grads).norm().item()
            torch.nn.utils.clip_grad_value_(self.parameters_evaluator, 1.0)
            self.optimizer_evaluator.step()

        ###### STEP 2: train Q with real transitions
        loss_Q = self.calculate_loss_TD(batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, type="kld")

        ###### STEP 3: train Q with imagined transitions (if evaluator is present, mask those that are delusional)
        loss_Q_dyna = self.calculate_loss_TD(batch_obs_curr, action_imagined, reward_imagined, obses_imagined, omegas_imagined, type="kld")
        if self.optimizer_evaluator is not None:
            with torch.no_grad():
                dist_distance2imagined = self.network_policy.network_evaluator(batch_obs_curr, obses_imagined, scalarize=False).softmax(-1).max(-2)[0]
                mask_rejected = dist_distance2imagined[:, 0] < 0.25 # arbitrary threshold
            loss_Q_dyna[mask_rejected.detach()] = 0.0
            if debug:
                writer.add_scalar("Debug/dist_distance2imagined_bin0", dist_distance2imagined[:, 0].mean().item(), self.steps_processed)
                # print("debug which ones are delusional, if we cannot do it in batches, we can do it in a loop or just a sample")
        else:
            mask_rejected = None

        ###### STEP 4: train generator
        loss_model, loss_recon, loss_reward, loss_omega, ratio_perfect_recon, diff_rewards, ratio_diff_omega, ratio_fp_omega, ratio_fn_omega = self.model.compute_loss(batch_processed, debug=debug)

        if self.prioritized_replay:
            assert weights is not None
            error_dyna_weighted = loss_model.mean() + ((loss_Q + loss_Q_dyna) * weights.detach()).mean() # kaixhin's rainbow implementation used mean()
        else:
            error_dyna_weighted = loss_model.mean() + (loss_Q + loss_Q_dyna).mean()
        self.optimizer_dyna.zero_grad(set_to_none=True)
        error_dyna_weighted.backward()

        if debug:
            with torch.no_grad():
                grads = [param.grad.detach().flatten() for param in self.parameters_dyna] # if param.grad is not None
                norm_grad_dyna = torch.cat(grads).norm().item()
        torch.nn.utils.clip_grad_value_(self.parameters_dyna, 1.0)
        self.optimizer_dyna.step()

        with torch.no_grad():
            if self.prioritized_replay or debug:
                res_TD = self.calculate_TD_L1_scalar(batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done)
            if self.prioritized_replay:
                priorities = (res_TD + 1e-6).squeeze().detach()
                self.hrb.update_priorities(batch_idxes, priorities.cpu().numpy())
                if debug: 
                    writer.add_scalar("Train/priorities", priorities.mean().item(), self.steps_processed)

            if debug:
                if self.prioritized_replay:
                    loss_overall_weighted = (loss_model * weights).detach().mean()
                    loss_recon_weighted = (loss_recon * weights).detach().mean()
                    loss_reward_weighted = (loss_reward * weights).detach().mean()
                    loss_omega_weighted = (loss_omega* weights).detach() .mean()
                else:
                    loss_overall_weighted = loss_model.detach().mean()
                    loss_recon_weighted = loss_recon.detach().mean()
                    loss_reward_weighted = loss_reward.detach().mean()
                    loss_omega_weighted = loss_omega.detach().mean()

                writer.add_scalar(f"Train_Dyna_Model/loss_overall", loss_overall_weighted.item(), self.steps_processed)
                writer.add_scalar(f"Train_Dyna_Model/loss_recon", loss_recon_weighted.item(), self.steps_processed)
                writer.add_scalar(f"Train_Dyna_Model/loss_reward", loss_reward_weighted.item(), self.steps_processed)
                writer.add_scalar(f"Train_Dyna_Model/loss_omega", loss_omega_weighted.item(), self.steps_processed)
                writer.add_scalar(f"Debug_Dyna_Model/ratio_imperfect_recon", 1 - ratio_perfect_recon.item(), self.steps_processed)
                writer.add_scalar(f"Debug_Dyna_Model/diff_rewards", diff_rewards.item(), self.steps_processed)
                writer.add_scalar(f"Debug_Dyna_Model/ratio_diff_omega", ratio_diff_omega.item(), self.steps_processed)
                if ratio_fp_omega is not None:
                    writer.add_scalar(f"Debug_Dyna_Model/ratio_fp_omega", ratio_fp_omega.item(), self.steps_processed)
                if ratio_fn_omega is not None:
                    writer.add_scalar(f"Debug_Dyna_Model/ratio_fn_omega", ratio_fn_omega.item(), self.steps_processed)

                writer.add_scalar("Debug/res_TD", res_TD.mean().item(), self.steps_processed)
                writer.add_scalar("Debug/norm_grad_dyna", norm_grad_dyna, self.steps_processed)
                writer.add_scalar("Train/loss_Q", loss_Q.mean().item(), self.steps_processed)
                if mask_rejected is None:
                    writer.add_scalar("Train/loss_Q_dyna", loss_Q_dyna.mean().item(), self.steps_processed)
                elif not mask_rejected.all():
                    writer.add_scalar("Train/loss_Q_dyna", loss_Q_dyna[~mask_rejected].mean().item(), self.steps_processed)
                writer.add_scalar("Train/loss_model", loss_model.mean().item(), self.steps_processed)

                if self.optimizer_evaluator is not None:
                    writer.add_scalar("Debug/norm_rep", torch.sqrt((batch_state_curr.flatten(1) ** 2).sum(-1)).mean().item(), self.steps_processed)
                    writer.add_scalar("Debug/norm_rep_local", torch.sqrt((state_local_curr**2).sum(-1)).mean().item(), self.steps_processed)
                    writer.add_scalar("Debug/norm_grad_evaluator", norm_grad_evaluator, self.steps_processed)
                    writer.add_scalar("Train/loss_discount", loss_discount.mean().item(), self.steps_processed)
                    writer.add_scalar("Debug/ratio_rejected", mask_rejected.float().mean().item(), self.steps_processed)
                else:
                    writer.add_scalar("Debug/ratio_rejected", 0.0, self.steps_processed)

    def sync_parameters(self):
        """
        synchronize the parameters of self.network_policy and self.network_target
        this is hard sync, maybe a softer version is going to do better
        model not synced, it doesn't need a target net for bootstrapping
        """
        self.network_target.encoder_Q.load_state_dict(self.network_policy.encoder_Q.state_dict())
        self.network_target.estimator_Q.load_state_dict(self.network_policy.estimator_Q.state_dict())
        if self.network_policy.encoder_evaluator is not None:
            self.network_target.encoder_evaluator.load_state_dict(self.network_policy.encoder_evaluator.state_dict())
        if self.network_policy.binder_evaluator is not None:
            self.network_target.binder_evaluator.load_state_dict(self.network_policy.binder_evaluator.state_dict())
        if self.network_policy.evaluator_discount is not None:
            self.network_target.evaluator_discount.load_state_dict(self.network_policy.evaluator_discount.state_dict())
        if not self.silent:
            print("policy-target parameters synced")

def create_DQN_Dyna_network(args, env, dim_embed, num_actions, device, share_memory=False):
    from models import Encoder_MiniGrid, Binder_MiniGrid, Predictor_MiniGrid

    if args.activation == "relu":
        activation = torch.nn.ReLU
    elif args.activation == "elu":
        activation = torch.nn.ELU
    elif args.activation == "leakyrelu":
        activation = torch.nn.LeakyReLU
    elif args.activation == "silu":
        activation = torch.nn.SiLU

    obs_sample = minigridobs2tensor(env.obs_curr)

    if args.dyna_reject:
        encoder_evaluator = Encoder_MiniGrid(dim_embed, obs_sample=env.reset(), norm=bool(args.layernorm), append_pos=bool(args.append_pos), activation=activation)
        encoder_evaluator.to(device)
        if share_memory:
            encoder_evaluator.share_memory()

        sample_staterep_evaluator = encoder_evaluator(obs_sample)

        binder_evaluator = Binder_MiniGrid(
            sample_staterep_evaluator,
            len_rep=args.len_rep,
            norm=bool(args.layernorm),
            activation=activation,
            num_heads=args.num_heads,
            size_bottleneck=args.size_bottleneck,
            type_arch=args.arch_enc,
        )
        binder_evaluator.to(device)
        if share_memory:
            binder_evaluator.share_memory()
    
        if args.transform_discount_target:
            dict_head_discount = {
                "len_predict": num_actions,
                "dist_out": True,
                "value_min": 1,
                "value_max": args.atoms_discount,
                "atoms": args.atoms_discount,
                "classify": False,
            }
        else:
            dict_head_discount = {
                "len_predict": num_actions,
                "dist_out": True,
                "value_min": 0,
                "value_max": args.gamma,
                "atoms": args.atoms_discount,
                "classify": False,
            }
        evaluator_discount = Predictor_MiniGrid(
            num_actions,
            len_input=binder_evaluator.len_out,
            depth=args.depth_hidden,
            width=args.width_hidden,
            norm=bool(args.layernorm),
            activation=activation,
            dict_head=dict_head_discount,
        )
        if args.transform_discount_target:
            evaluator_discount.histogram_converter.support_distance = torch.arange(1, args.atoms_discount + 1, device=device, dtype=torch.float32)
            evaluator_discount.histogram_converter.support_discount = torch.pow(args.gamma, evaluator_discount.histogram_converter.support_distance)
            # evaluator_discount.histogram_converter.support_discount[-1] = 0.0
        else:
            evaluator_discount.histogram_converter.support_discount = torch.linspace(0, args.gamma, args.atoms_discount, device=device, dtype=torch.float32)
            evaluator_discount.histogram_converter.support_distance = torch.log(evaluator_discount.histogram_converter.support_discount) / np.log(args.gamma)
            evaluator_discount.histogram_converter.support_distance.clamp_(1, 250)
        evaluator_discount.histogram_converter.support_override = True
        evaluator_discount.to(device)
        if share_memory:
            evaluator_discount.share_memory()
    else:
        encoder_evaluator, binder_evaluator, evaluator_discount = None, None, None

    encoder_Q = torch.nn.Sequential(
        Encoder_MiniGrid(args.dim_embed, obs_sample, norm=True, append_pos=False, activation=activation),
        activation(False),
        torch.nn.Flatten(),
        torch.nn.Linear(args.dim_embed * obs_sample.shape[-2] * obs_sample.shape[-3], args.len_rep),
    )
    estimator_Q = Predictor_MiniGrid(env.action_space.n, args.len_rep, depth=args.depth_hidden, width=args.width_hidden, activation=activation, norm=bool(args.layernorm), dict_head={"len_predict": None, "dist_out": True, "value_min": args.value_min, "value_max": args.value_max, "atoms": args.atoms_value, "classify": False}, value_min=args.value_min, value_max=args.value_max)
    estimator_Q.to(device)
    if share_memory:
        estimator_Q.share_memory()

    if "RandDistShift" in args.game:
        from models import Encoder_MiniGrid_RDS, Decoder_MiniGrid_RDS
        encoder_CVAE = Encoder_MiniGrid_RDS()
        decoder_CVAE = Decoder_MiniGrid_RDS()
        num_classes_abstract = 1
    elif "SwordShieldMonster" in args.game:
        from models import Encoder_MiniGrid_SSM, Decoder_MiniGrid_SSM
        encoder_CVAE = Encoder_MiniGrid_SSM()
        decoder_CVAE = Decoder_MiniGrid_SSM()
        num_classes_abstract = 4
    else:
        raise NotImplementedError()
    
    len_action = 16
    argmax_reconstruction = True

    model = MINIGRID_OBS_ONESTEP_MODEL(
        encoder_CVAE,
        decoder_CVAE,
        obs_sample,
        num_actions=env.action_space.n,
        len_action=len_action,
        dim_embed_bow=args.dim_embed,
        activation=activation,
        argmax_reconstruction=argmax_reconstruction,
        num_classes_abstract=num_classes_abstract,
        prob_random_generation=0.5,
        len_rep=args.len_rep,
    )
    model.to(device)
    if share_memory:
        model.share_memory()

    network_policy = DQN_DYNA_NETWORK(encoder_Q=encoder_Q, estimator_Q=estimator_Q, encoder_evaluator=encoder_evaluator, binder_evaluator=binder_evaluator, evaluator_discount=evaluator_discount, model=model)
    network_policy.to(device)
    if share_memory:
        network_policy.share_memory()
    return network_policy


def create_DQN_Dyna_agent(
    args, env, dim_embed, num_actions, device=None, hrb=None, network_policy=None, network_target=None, inference_only=False, silent=False
):
    if device is None:
        if torch.cuda.is_available() and not args.force_cpu:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            warnings.warn("agent created on cpu")

    if not inference_only and hrb is None:
        from utils import get_cpprb

        hrb = get_cpprb(
            env,
            args.size_buffer,
            prioritized=args.prioritized_replay,
            hindsight=True,
            hindsight_strategy=args.hindsight_strategy,
            num_envs=args.num_envs_train,
        )

    if network_policy is None:
        network_policy = create_DQN_Dyna_network(args, env, dim_embed, num_actions, device=device, share_memory=False)

    if inference_only:
        agent = DQN_DYNA_BASE(
            env,
            network_policy,
            gamma=args.gamma,
            steps_total=args.steps_max,
            prioritized_replay=bool(args.prioritized_replay),
            device=device,
            seed=args.seed,
            hrb=hrb,
            silent=silent,
            transform_discount_target=args.transform_discount_target,
            prob_relabel_generateJIT=args.prob_relabel_generateJIT,
        )
    else:
        agent = DQN_DYNA(
            env,
            network_policy,
            gamma=args.gamma,
            steps_total=args.steps_max,
            prioritized_replay=bool(args.prioritized_replay),
            freq_train=args.freq_train,
            freq_targetsync=args.freq_targetsync,
            lr=args.lr,
            size_batch=args.size_batch,
            device=device,
            seed=args.seed,
            hrb=hrb,
            silent=silent,
            transform_discount_target=args.transform_discount_target,
            prob_relabel_generateJIT=args.prob_relabel_generateJIT,
        )
    return agent
