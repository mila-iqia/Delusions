import torch, numpy as np, copy, os
import warnings
from utils import LinearSchedule, minigridobs2tensor, RL_AGENT

from utils import abstract_planning, generate_random_waypoints, append_GT_graph, k_medoids, find_unique, process_batch, reachability_from_distances, take_submatrix
from visual_utils import visualize_waypoint_graph, visualize_plan


class DQN_SKIPPER_NETWORK(torch.nn.Module):
    def __init__(self, encoder, binder, estimator_Q, estimator_discount, estimator_reward, estimator_omega, cvae=None):
        super(DQN_SKIPPER_NETWORK, self).__init__()
        self.encoder = encoder
        self.binder = binder
        self.estimator_Q = estimator_Q
        self.estimator_discount = estimator_discount
        self.estimator_reward = estimator_reward
        self.estimator_omega = estimator_omega
        self.cvae = cvae

    def to(self, device):
        super().to(device)
        self.encoder.to(device)
        self.binder.to(device)
        if self.estimator_Q is not None:
            self.estimator_Q.to(device)
        self.estimator_discount.to(device)
        self.estimator_reward.to(device)
        self.estimator_omega.to(device)
        if self.cvae is not None:
            self.cvae.to(device)

    def parameters(self):
        parameters = []
        parameters += list(self.encoder.parameters())
        parameters += list(self.binder.parameters())
        if self.estimator_Q is not None:
            parameters += list(self.estimator_Q.parameters())
        parameters += list(self.estimator_discount.parameters())
        parameters += list(self.estimator_reward.parameters())
        parameters += list(self.estimator_omega.parameters())
        if self.cvae is not None:
            parameters += list(self.cvae.parameters())
        return parameters

class DQN_SKIPPER_BASE(RL_AGENT):
    def __init__(
        self,
        env,
        network_policy,
        freq_plan=16,
        num_waypoints=16,
        waypoint_strategy="once",
        always_select_goal=False,
        optimal_plan=False,
        optimal_policy=False,
        dist_cutoff=8,
        gamma=0.99,
        gamma_int=0.95,
        type_intrinsic_reward="sparse",
        clip_reward=True,
        exploration_fraction=0.02,
        epsilon_final_train=0.01,
        epsilon_eval=0.001,
        steps_total=50000000,
        prioritized_replay=True,
        func_obs2tensor=minigridobs2tensor,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=42,
        valid_waypoints_only=False,
        no_lava_waypoints=False,
        hrb=None,
        silent=False,
        transform_discount_target=True,
        num_waypoints_unpruned=32,
        prob_relabel_generateJIT=0.0,
        no_Q_head=False,
        unique_codes=False,
        unique_obses=True,
        nonsingleton=False,
    ):
        super(DQN_SKIPPER_BASE, self).__init__(env, gamma, seed)

        self.clip_reward = clip_reward
        self.schedule_epsilon = LinearSchedule(
            schedule_timesteps=int(exploration_fraction * steps_total),
            initial_p=1.0,
            final_p=epsilon_final_train,
        )
        self.epsilon_eval = epsilon_eval

        self.gamma_int = gamma_int
        self.type_intrinsic_reward = type_intrinsic_reward
        self.nonsingleton = bool(nonsingleton)

        self.device = device
        self.always_select_goal = bool(always_select_goal)
        self.optimal_plan = bool(optimal_plan)
        self.optimal_policy = bool(optimal_policy)

        self.freq_plan, self.step_last_planned = freq_plan, 0
        self.num_waypoints = num_waypoints
        assert waypoint_strategy in ["once", "regenerate_whole_graph", "grow"]
        self.waypoint_strategy = waypoint_strategy
        self.num_waypoints_unpruned = num_waypoints_unpruned
        assert self.num_waypoints_unpruned >= self.num_waypoints

        self.network_policy = network_policy
        self.network_target = self.network_policy

        self.support_discount = self.network_policy.estimator_discount.histogram_converter.support_discount
        self.support_distance = self.network_policy.estimator_discount.histogram_converter.support_distance
        self.support_reward = self.network_policy.estimator_reward.histogram_converter.support
        self.cvae = self.network_policy.cvae

        if self.cvae is None:
            if env.name_game == "RandDistShift":
                self.encoder_wp = lambda obs, env: np.array(env.obs2ijd(obs))
                self.decoder_wp = lambda ijd, env: env.ijd2obs(*ijd)
            elif env.name_game == "SwordShieldMonster":
                self.encoder_wp = lambda obs, env: np.array(env.obs2ijxd(obs))
                self.decoder_wp = lambda ijxd, env: env.ijxd2obs(*ijxd)
        else:
            self.encoder_wp = lambda obs: self.cvae.encode_from_obs(obs).reshape(obs.shape[0], -1).squeeze_().cpu().numpy()
            self.decoder_wp = lambda code, obs: self.cvae.decode_to_obs(code, obs)
        self.prob_relabel_generateJIT = float(prob_relabel_generateJIT)

        self.valid_waypoints_only = bool(valid_waypoints_only)
        self.no_lava_waypoints = bool(no_lava_waypoints)

        self.transform_discount_target = bool(transform_discount_target)
        self.dist_cutoff = dist_cutoff

        self.steps_interact, self.steps_total = 0, steps_total
        self.steps_processed = 0

        self.step_last_print, self.time_last_print = 0, None

        self.obs2tensor = lambda obs: func_obs2tensor(obs, device=self.device)

        self.prioritized_replay = prioritized_replay
        self.hrb = hrb
        if self.prioritized_replay:
            self.schedule_beta_sample_priorities = LinearSchedule(steps_total, initial_p=0.4, final_p=1.0)
        self.silent = silent

        self.waypoints_existing, self.proxy_graph_curr = None, None

        self.no_Q_head = bool(no_Q_head)
        self.unique_codes = bool(unique_codes)
        self.unique_obses = bool(unique_obses)

        self.on_episode_end(eval=True)

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
    def another_waypoint_reached(self, obs_curr, env, writer=None, step_record=None):
        if self.waypoints_existing is None:
            return False
        if self.waypoint_curr is None:
            if self.cvae is None:
                self.waypoint_curr = self.encoder_wp(obs_curr, env)
            else:
                self.waypoint_curr = self.encoder_wp(self.obs2tensor(obs_curr))
        if self.waypoint_targ is not None:
            if (self.waypoint_curr == self.waypoint_targ).all():
                self.waypoint_last_reached = copy.copy(self.waypoint_curr)
                self.idx_wp_last_reached = int(self.idx_waypoint_targ)
                self.num_waypoints_reached += 1
                if not self.silent:
                    print(f"planning triggered at step {self.steps_interact:d}: waypoint_targ {self.waypoint_targ.tolist()} reached")
                self.waypoint_targ, self.state_wp_targ, self.idx_waypoint_targ = None, None, None
                return True
        coincidence = (self.waypoints_existing == self.waypoint_curr).all(-1)
        if self.waypoint_last_reached is not None:
            coincidence &= (self.waypoints_existing != self.waypoint_last_reached).any(-1)
        found = coincidence.any()
        if found:
            self.waypoint_last_reached = copy.copy(self.waypoint_curr)
            self.idx_wp_last_reached = np.where(coincidence)[0][0]
            self.num_waypoints_reached += 1
            if not self.silent:
                print(
                    f"planning triggered at step {self.steps_interact:d}: unexpected waypoint {self.waypoint_curr.tolist()} reached",
                    end="\n" if self.waypoint_targ is None else "",
                )
                if self.waypoint_targ is not None:
                    print(f", instead of {self.waypoint_targ.tolist()}")
        return found

    def Q_conditioned(self, batch_curr, waypoint_targ=None, type_curr="obs", env=None, obs_targ=None):
        """
        fast forward pass for conditioned Q
        """
        assert waypoint_targ is not None or obs_targ is not None
        if obs_targ is None:
            if self.cvae is None:
                if self.obs_wp_targ is None:
                    self.obs_wp_targ = self.obs2tensor(self.decoder_wp(waypoint_targ, env))
                obs_targ = self.obs_wp_targ
            else:
                obs_targ = self.obs2tensor(self.decoder_wp(waypoint_targ, env))
        elif isinstance(obs_targ, np.ndarray):
            obs_targ = self.obs2tensor(obs_targ)
        state_targ = self.network_policy.encoder(obs_targ)
        if type_curr == "obs":
            if isinstance(batch_curr, np.ndarray):
                batch_obs_curr = self.obs2tensor(batch_curr)
            else:
                batch_obs_curr = batch_curr
            state_curr = self.network_policy.encoder(batch_obs_curr)
        elif type_curr == "state_rep":
            state_curr = batch_curr
        if state_curr.shape[0] > 1 and state_targ.shape[0] == 1:
            state_targ = state_targ.expand_as(state_curr)
        state_local_binded = self.network_policy.binder(state_curr, state_targ)
        if self.no_Q_head:
            dist_discounts = self.network_policy.estimator_discount(state_local_binded, scalarize=False).softmax(-1)
            return dist_discounts @ self.support_discount
        else:
            return self.network_policy.estimator_Q(state_local_binded, scalarize=True)

    @torch.no_grad()
    def reinit_plan(self):
        self.waypoint_last_reached = None
        self.idx_wp_last_reached = None
        self.idx_waypoint_targ = None
        self.waypoint_targ = None
        self.waypoint_goal = None
        self.obs_goal_tensor = None
        self.state_wp_targ = None
        self.ijxd_targ = None
        self.obs_wp_targ = None
        self.replan = True

    @torch.no_grad()
    def on_episode_end(self, eval=False):
        if self.optimal_policy:
            self.Q_oracle, self.ijxd_targ_oracle = None, None
        self.reinit_plan()
        self.waypoints_existing = None
        self.replan = True
        try:
            if self.proxy_graph_curr is not None:
                del self.proxy_graph_curr
                self.proxy_graph_curr = None
        except:
            self.proxy_graph_curr = None
        try:
            if self.vertices_unpruned is not None:
                del self.vertices_unpruned
                self.vertices_unpruned = None
        except:
            self.vertices_unpruned = None
        try:
            if self.obses_wps_existing is not None:
                del self.obses_wps_existing
                self.obses_wps_existing = None
        except:
            self.obses_wps_existing = None
        self.num_planning_triggered = 0
        self.num_planning_triggered_timeout = 0
        self.num_waypoints_reached = 0
        self.code_goal = None
        if self.hrb is not None and not eval:
            self.hrb.on_episode_end()

    def check_if_target_reached(self, obs, obs_targ):
        batch_targ_really_reached = (obs == obs_targ).reshape(obs_targ.shape[0], -1).all(-1)
        if self.nonsingleton:
            obs_np, obs_targ_np = obs.cpu().numpy(), obs_targ.cpu().numpy()
            if self.env.name_game == 'RandDistShift':
                i_next, j_next = self.env.obs2ijd(obs_np)
                i_targ, j_targ = self.env.obs2ijd(obs_targ_np)
                batch_targ_reached = (np.abs(i_next - i_targ) + np.abs(j_next - j_targ)) <= 1
            elif self.env.name_game == 'SwordShieldMonster':
                i_next, j_next, x_next = self.env.obs2ijxd(obs_np)
                i_targ, j_targ, x_targ = self.env.obs2ijxd(obs_targ_np)
                batch_targ_reached = ((np.abs(i_next - i_targ) + np.abs(j_next - j_targ)) <= 1) & (x_next == x_targ)
            else:
                raise NotImplementedError
            batch_targ_reached = torch.tensor(batch_targ_reached, device=obs.device, dtype=torch.bool)
        else:
            batch_targ_reached = batch_targ_really_reached
        return batch_targ_really_reached, batch_targ_reached

    def calculate_multihead_error(
        self,
        batch_obs_curr,
        batch_action,
        batch_reward,
        batch_obs_next,
        batch_done,
        batch_obs_targ,
        batch_reward_int=None,
        calculate_Q_error=True,
        calculate_reward_error=True,
        calculate_omega_error=True,
        calculate_priorities=True,
        freeze_encoder=False,
        freeze_binder=False,
        type_priorities="kl",
        state_local_curr=None,
        state_local_next=None,
        state_local_next_targetnet=None,
    ):
        size_batch = batch_obs_curr.shape[0]

        if state_local_curr is not None or state_local_next is not None or state_local_next_targetnet is not None:
            assert state_local_curr is not None and state_local_next is not None and state_local_next_targetnet is not None
            batch_state_curr = None
            flag_reuse = True
        else:
            flag_reuse = False

        with torch.no_grad():
            batch_targ_really_reached, batch_targ_reached = self.check_if_target_reached(batch_obs_next, batch_obs_targ)
            batch_done_augmented = torch.logical_or(batch_targ_reached, batch_done)
            if not flag_reuse:
                batch_obs_next_targ = torch.cat([batch_obs_next, batch_obs_targ], 0)
                batch_obs_curr_next_targ = torch.cat([batch_obs_curr, batch_obs_next_targ], 0)

        with torch.set_grad_enabled(not freeze_encoder):
            if flag_reuse:
                batch_state_targ = self.network_policy.encoder(batch_obs_targ)
            else:
                batch_state_curr_next_targ = self.network_policy.encoder(batch_obs_curr_next_targ)
                batch_state_curr, batch_state_next, batch_state_targ = batch_state_curr_next_targ.chunk(3, dim=0)

        with torch.set_grad_enabled(not freeze_binder):
            if flag_reuse:
                if self.network_policy.binder.local_perception:
                    state_local_targ = self.network_policy.binder.extract_local_field(batch_state_targ)
                else:
                    state_local_targ = self.network_policy.binder.flattener(batch_state_targ)
                states_local_curr_targ = torch.cat([state_local_curr, state_local_targ], -1)
            else:
                if self.network_policy.binder.local_perception:
                    state_local_curr_next_targ = self.network_policy.binder.extract_local_field(batch_state_curr_next_targ)
                else:
                    state_local_curr_next_targ = self.network_policy.binder.flattener(batch_state_curr_next_targ)
                state_local_curr, state_local_next, state_local_targ = torch.split(state_local_curr_next_targ, [size_batch, size_batch, size_batch], dim=0)
                states_local_curr_targ = torch.cat([state_local_curr, state_local_targ], -1)

        predicted_discount = self.network_policy.estimator_discount(states_local_curr_targ, batch_action, scalarize=False)

        with torch.no_grad():
            states_local_next_targ = torch.cat([state_local_next.detach(), state_local_targ.detach()], -1)
            if self.no_Q_head:
                softmax_predicted_discount_next = self.network_policy.estimator_discount(states_local_next_targ.detach(), scalarize=False).softmax(-1)
                predicted_discount_next = softmax_predicted_discount_next @ self.support_discount
                action_next = torch.argmax(predicted_discount_next.detach(), dim=1, keepdim=True)
            else:
                predicted_Q_next = self.network_policy.estimator_Q(states_local_next_targ.detach(), scalarize=True)
                action_next = torch.argmax(predicted_Q_next.detach(), dim=1, keepdim=True)
            if flag_reuse:
                batch_state_targ_targetnet = self.network_target.encoder(batch_obs_targ)
                if self.network_target.binder.local_perception:
                    state_local_targ_targetnet = self.network_policy.binder.extract_local_field(batch_state_targ_targetnet)
                else:
                    state_local_targ_targetnet = self.network_policy.binder.flattener(batch_state_targ_targetnet)
            else:
                batch_state_next_targ_targetnet = self.network_target.encoder(batch_obs_next_targ)
                if self.network_target.binder.local_perception:
                    state_local_next_targ_targetnet = self.network_policy.binder.extract_local_field(batch_state_next_targ_targetnet)
                else:
                    state_local_next_targ_targetnet = self.network_policy.binder.flattener(batch_state_next_targ_targetnet)
                state_local_next_targetnet, state_local_targ_targetnet = torch.split(state_local_next_targ_targetnet, [size_batch, size_batch], dim=0)
            states_local_next_targ_targetnet = torch.cat([state_local_next_targetnet, state_local_targ_targetnet], -1)

        with torch.no_grad():
            dist_discounts = self.network_target.estimator_discount(states_local_next_targ_targetnet, action_next, scalarize=False).softmax(-1)
            if self.transform_discount_target:
                distance_next = (dist_discounts @ self.support_distance).reshape(size_batch, 1)
                distance_next[batch_done] = 1000.0
                distance_next[batch_targ_reached] = 0.0
                target_discount_distance = 1.0 + distance_next
            else:
                discount_next = (dist_discounts @ self.network_target.estimator_discount.histogram_converter.support_discount).reshape(size_batch, 1)
                discount_next[batch_done] = 0.0
                discount_next[batch_targ_reached] = 1.0
                target_discount_distance = self.gamma * discount_next
            target_discount_dist = self.network_target.estimator_discount.histogram_converter.to_histogram(target_discount_distance)
        discount_logits_curr = predicted_discount.reshape(size_batch, -1)
        loss_discount = torch.nn.functional.kl_div(torch.log_softmax(discount_logits_curr, -1), target_discount_dist.detach(), reduction="none").sum(-1)

        if calculate_Q_error and not self.no_Q_head:
            predicted_Q = self.network_policy.estimator_Q(states_local_curr_targ, batch_action, scalarize=False)
            with torch.no_grad():
                values_next = self.network_target.estimator_Q(states_local_next_targ_targetnet, action=action_next, scalarize=True).reshape(size_batch, -1)
                if self.type_intrinsic_reward == "sparse":
                    batch_reward_int = batch_targ_really_reached.float().reshape(size_batch, -1) if batch_reward_int is None else batch_reward_int
                    values_next[batch_done_augmented] = 0
                elif self.type_intrinsic_reward == "dense":
                    batch_reward_int = torch.full_like(batch_reward, -1) if batch_reward_int is None else batch_reward_int
                    values_next[batch_done] = -1000
                    values_next[batch_targ_really_reached] = 0
                else:
                    raise NotImplementedError()
                target_Q = batch_reward_int + self.gamma_int * values_next
                Q_dist_target = self.network_target.estimator_Q.histogram_converter.to_histogram(target_Q)
            Q_logits_curr = predicted_Q.reshape(size_batch, -1)
            loss_TD = torch.nn.functional.kl_div(torch.log_softmax(Q_logits_curr, -1), Q_dist_target.detach(), reduction="none").sum(-1)
        else:
            loss_TD = torch.zeros_like(loss_discount)

        if calculate_reward_error:
            logits_reward_curr = self.network_policy.estimator_reward(states_local_curr_targ, batch_action, scalarize=False)
            with torch.no_grad():
                G_next = self.network_target.estimator_reward(states_local_next_targ_targetnet, action=action_next, scalarize=True).reshape(size_batch, -1)
                G_next[batch_done_augmented] = 0.0
                target_G = batch_reward + self.gamma * G_next
                G_dist_target = self.network_target.estimator_reward.histogram_converter.to_histogram(target_G)
            G_logits_curr = logits_reward_curr.reshape(size_batch, -1)
            loss_reward = torch.nn.functional.kl_div(torch.log_softmax(G_logits_curr, -1), G_dist_target.detach(), reduction="none").sum(-1)
        else:
            loss_reward = torch.zeros_like(loss_discount)

        if calculate_omega_error:
            predicted_omega = self.network_policy.estimator_omega(state_local_next, scalarize=False)
            omega_logits_pred = predicted_omega.reshape(-1, 2)
            loss_omega = torch.nn.functional.cross_entropy(torch.log_softmax(omega_logits_pred, -1), batch_done.to(torch.long).detach(), reduction="none")
        else:
            omega_logits_pred = None
            loss_omega = torch.zeros_like(loss_TD)
        if calculate_priorities:
            with torch.no_grad():
                if type_priorities == "kanto":
                    kanto_discount = (target_discount_dist - discount_logits_curr.softmax(-1)).abs_().sum(-1)
                    if not calculate_reward_error:
                        kanto_reward = torch.zeros_like(kanto_discount)
                    else:
                        kanto_reward = (G_dist_target - G_logits_curr.softmax(-1)).abs_().sum(-1)
                    if not calculate_Q_error or self.no_Q_head:
                        kanto_Q = torch.zeros_like(kanto_discount)
                    else:
                        kanto_Q = (Q_dist_target - Q_logits_curr.softmax(-1)).abs_().sum(-1)
                    priorities = 0.5 * (kanto_Q + kanto_discount + kanto_reward).detach()
                elif type_priorities == "kl":
                    priorities = (loss_TD + loss_discount + loss_reward + loss_omega).squeeze().detach() * 0.25
                elif type_priorities == "abs_TD":
                    assert calculate_Q_error and not self.no_Q_head and self.type_intrinsic_reward == "sparse"
                    Q_curr = self.network_policy.estimator_Q.histogram_converter.from_histogram(Q_logits_curr, logits=True)
                    priorities = (target_Q.squeeze() - Q_curr.squeeze()).abs().detach()
                else:
                    raise NotImplementedError()
        else:
            priorities = None
        return priorities, loss_TD, loss_discount, loss_reward, loss_omega, omega_logits_pred, batch_state_curr, state_local_curr, state_local_next, state_local_next_targetnet

    @torch.no_grad()
    def get_abstract_graph(self, dict_waypoints, obs_curr=None, env=None, save_wp_existing_obses=False):
        if isinstance(obs_curr, np.ndarray):
            obs_curr = self.obs2tensor(obs_curr)
        if self.obses_wps_existing is None:
            if self.cvae is None:
                waypoints_existing = dict_waypoints["ijxds"]
                assert env is not None
                wp_existing_obses = self.obs2tensor(self.decoder_wp(np.split(waypoints_existing, waypoints_existing.shape[1], axis=1), env))
            else:
                wp_existing_obses = self.obs2tensor(dict_waypoints["obses"])
            if save_wp_existing_obses:
                self.obses_wps_existing = wp_existing_obses
        else:
            wp_existing_obses = self.obses_wps_existing
        if obs_curr is None:
            wp_obses = wp_existing_obses
        else:
            wp_obses = torch.cat([obs_curr.reshape(1, *wp_existing_obses.shape[1:]), wp_existing_obses], dim=0)
        return self.edges_from_vertices(wp_obses)

    def edges_from_vertices(self, obses):
        num_waypoints = obses.shape[0]
        wp_states = self.network_policy.encoder(obses)
        if self.network_policy.binder.local_perception:
            wp_states_local = self.network_policy.binder.extract_local_field(wp_states)
        else:
            wp_states_local = self.network_policy.binder.flattener(wp_states)
        tuples = torch.cat([torch.repeat_interleave(wp_states_local, num_waypoints, dim=0), wp_states_local.repeat([num_waypoints, 1])], -1)
        omegas = self.network_policy.estimator_omega(wp_states_local, scalarize=True).bool().squeeze()
        if self.no_Q_head:
            softmax_discount_dist = self.network_policy.estimator_discount(tuples, scalarize=False).softmax(-1)
            predicted_discounts = softmax_discount_dist @ self.support_discount
            actions_greedy = torch.argmax(predicted_discounts, dim=1, keepdim=True)
            discounts = predicted_discounts.gather(1, actions_greedy).reshape(num_waypoints, num_waypoints)
            dist_discounts = softmax_discount_dist[
                torch.arange(softmax_discount_dist.shape[0], device=softmax_discount_dist.device),
                actions_greedy.squeeze(),
            ]
        else:
            predicted_Q = self.network_policy.estimator_Q(tuples, scalarize=True)
            actions_greedy = torch.argmax(predicted_Q, dim=1, keepdim=True)
            dist_discounts = self.network_policy.estimator_discount(tuples, actions_greedy, scalarize=False).softmax(-1)
            discounts = (dist_discounts @ self.support_discount).reshape(num_waypoints, num_waypoints)
        distances = (dist_discounts @ self.support_distance).reshape(num_waypoints, num_waypoints)
        rewards = self.network_policy.estimator_reward(tuples, actions_greedy, scalarize=True).reshape(num_waypoints, num_waypoints)
        return dict(discounts=discounts, distances=distances, rewards=rewards, omegas=omegas, Q=None)

    def get_random_action(self, trigger_replan=False):
        if trigger_replan:
            self.replan = True
        return self.action_space.sample()

    @torch.no_grad()
    def prune_proxy_with_indices(self, vertices=None, edges=None, indices=None):
        assert indices is not None
        assert vertices is not None or edges is not None
        if vertices is not None:
            indices_cpu = indices if isinstance(indices, list) else None
            if "ijxds" in vertices:
                if indices_cpu is None:
                    indices_cpu = indices.cpu().numpy()
                vertices["ijxds"] = vertices["ijxds"][indices_cpu, :]
            if "states" in vertices:
                if indices_cpu is None:
                    indices_cpu = indices.cpu().numpy()
                vertices["states"] = vertices["states"][indices_cpu]
            if "mask_nonexistent" in vertices:
                if indices_cpu is None:
                    indices_cpu = indices.cpu().numpy()
                vertices["mask_nonexistent"] = vertices["mask_nonexistent"][indices_cpu]
            if "mask_irreversible" in vertices:
                if indices_cpu is None:
                    indices_cpu = indices.cpu().numpy()
                vertices["mask_irreversible"] = vertices["mask_irreversible"][indices_cpu]
        if edges is not None:
            edges["distances"], mask_chosen = take_submatrix(edges["distances"], indices=indices, return_mask2d=True)
            edges["discounts"] = take_submatrix(edges["discounts"], mask2d=mask_chosen)
            edges["rewards"] = take_submatrix(edges["rewards"], mask2d=mask_chosen)
            if edges["omegas"] is not None:
                edges["omegas"] = edges["omegas"][indices]
        return vertices, edges
    
    @torch.no_grad()
    def visualize_proxy(self, ijxds_highlight, wp_graph, env, obs_curr, writer, step_record, prefix_vis, suffix_vis=""):
        if env.name_game == "RandDistShift":
            rendered = env.render_image(ijs=ijxds_highlight[:, :2], obs=obs_curr)
        elif env.name_game == "SwordShieldMonster":
            i_curr, j_curr, x_curr = env.obs2ijxd(obs_curr)
            rendered = []
            for x in range(4):
                mask_this_image = (ijxds_highlight[:, 2] == x)
                obs_base = env.ijxd2obs(i_curr, j_curr, x)
                if x == x_curr:
                    _rendered = env.render_image(ijs=ijxds_highlight[mask_this_image, :2], obs=obs_base, agent_pos=(i_curr, j_curr), dir_agent=0)
                else:
                    _rendered = env.render_image(ijs=ijxds_highlight[mask_this_image, :2], obs=obs_base, no_agent=True)
                _rendered[:, :2, :] = 255
                _rendered[:, -2:, :] = 255
                _rendered[:2, :, :] = 255
                _rendered[-2:, :, :] = 255
                rendered.append(_rendered)
            rendered = np.concatenate(rendered, 1)
        if "Q" in wp_graph.keys():
            img_plan = visualize_plan(rendered, wp_graph, wp_graph["Q"], env, alpha=0.5)
            writer.add_image(f"{prefix_vis}/plan", img_plan, step_record, dataformats="HWC")
            img_Q = visualize_waypoint_graph(rendered, wp_graph, env, annotation="Q")
            writer.add_image(f"{prefix_vis}/Q", img_Q, step_record, dataformats="HWC")
        img_distances_plan = visualize_waypoint_graph(rendered, wp_graph, env, annotation="distances")
        writer.add_image(f"{prefix_vis}/distances{suffix_vis}", img_distances_plan, step_record, dataformats="HWC")
        img_discounts_plan = visualize_waypoint_graph(rendered, wp_graph, env, annotation="discounts")
        writer.add_image(f"{prefix_vis}/discounts{suffix_vis}", img_discounts_plan, step_record, dataformats="HWC")
        img_rewards_plan = visualize_waypoint_graph(rendered, wp_graph, env, annotation="rewards")
        writer.add_image(f"{prefix_vis}/rewards{suffix_vis}", img_rewards_plan, step_record, dataformats="HWC")

    @torch.no_grad()
    def decide(self, obs_curr, epsilon=None, eval=False, env=None, writer=None, random_walk=False, step_record=None):
        debug = writer is not None and self.num_planning_triggered == 0
        if epsilon is None:
            epsilon = self.epsilon_eval if eval else self.schedule_epsilon.value(self.steps_interact)
        else:
            assert epsilon >= 0 and epsilon <= 1.0
        if np.random.rand() < epsilon or (random_walk and not debug):
            return self.get_random_action()
        debug_visualize = debug if eval else debug and np.random.rand() < 0.01
        if debug:
            if eval:
                prefix_plan, prefix_debug, prefix_vis = "Plan_Eval", "Debug_Eval", "Visualize_Eval"
            else:
                prefix_plan, prefix_debug, prefix_vis = "Plan", "Debug", "Visualize"
            step_record = self.steps_interact if step_record is None else step_record
        obs_curr_tensor = None
        generate_graph = self.waypoints_existing is None or self.waypoint_strategy == "regenerate_whole_graph"
        self.waypoint_curr = None

        if self.replan:
            pass
        elif self.obs_wp_targ is None:
            self.replan = True
        elif generate_graph:
            self.replan = True
        elif self.steps_interact - self.step_last_planned >= self.freq_plan:
            self.replan = True
            self.num_planning_triggered_timeout += 1
        elif self.another_waypoint_reached(obs_curr, env, writer=writer, step_record=step_record):
            self.replan = True
        if self.replan:
            self.num_planning_triggered += 1
            self.replan = False
            self.step_last_planned = self.steps_interact
            if generate_graph:
                self.reinit_plan()
                self.obses_wps_existing = None
                if self.cvae is None:
                    vertices_unpruned = generate_random_waypoints(env, self.num_waypoints_unpruned, generate_DP_info=False, render=debug_visualize, valid_only=self.valid_waypoints_only, no_lava=self.no_lava_waypoints, return_dist=False, return_obs=True, unique=False, obs_curr=obs_curr)
                else:
                    if obs_curr_tensor is None:
                        obs_curr_tensor = self.obs2tensor(obs_curr)
                    if self.waypoint_curr is None:
                        self.waypoint_curr = self.encoder_wp(obs_curr_tensor)
                    if self.obs_goal_tensor is None:
                        self.obs_goal_tensor = self.obs2tensor(env.obs_goal)
                    if self.waypoint_goal is None:
                        self.waypoint_goal = self.cvae.encode_from_obs(self.obs_goal_tensor).reshape(1, -1)
                    codes_pred_tensor, obses_pred_tensor = self.cvae.generate_from_obs(obs_curr_tensor, num_samples=self.num_waypoints_unpruned - 2)
                    assert obses_pred_tensor.dtype == torch.uint8

                    codes_pred = torch.cat([torch.tensor(self.waypoint_curr[None, :], device=codes_pred_tensor.device, dtype=codes_pred_tensor.dtype), codes_pred_tensor.flatten(1), self.waypoint_goal], 0)
                    obses_pred = torch.cat([obs_curr_tensor, obses_pred_tensor, self.obs_goal_tensor], 0)

                    vertices_unpruned = {}
                    if self.optimal_plan or debug:
                        obses_pred_np = obses_pred.cpu().numpy()
                        if env.name_game == "RandDistShift":
                            states, ijxds = env.obs2ijdstate(obses_pred_np)
                        elif env.name_game == "SwordShieldMonster":
                            states, ijxds = env.obs2ijxdstate(obses_pred_np)
                        ijxds = np.stack(ijxds[: len(ijxds) - int(env.ignore_dir)], 1)
                        vertices_unpruned.update(ijxds=ijxds, states=states)
                    vertices_unpruned.update(obses=obses_pred, codes=codes_pred)

                if debug:
                    mask_wps_unpruned_nonexistent = np.zeros(self.num_waypoints_unpruned, dtype=bool)
                    if env.name_game == "SwordShieldMonster":
                        mask_wps_unpruned_irreversible = np.zeros(self.num_waypoints_unpruned, dtype=bool)
                        x_curr = int(vertices_unpruned["ijxds"][0][2])
                    for idx_state in range(self.num_waypoints_unpruned):
                        state_reachable_frominit = vertices_unpruned["states"][idx_state] in env.DP_info["states_reachable"]
                        mask_wps_unpruned_nonexistent[idx_state] = not state_reachable_frominit
                        if self.nonsingleton:
                            i_state, j_state = vertices_unpruned["ijxds"][idx_state][0], vertices_unpruned["ijxds"][idx_state][1]
                            if env.name_game == "SwordShieldMonster":
                                x_state = vertices_unpruned["ijxds"][idx_state][2]
                                if i_state > 0:
                                    mask_wps_unpruned_nonexistent[idx_state] &= not env.ijxd2state(i_state - 1, j_state, x_state) in env.DP_info["states_reachable"]
                                if i_state < env.width - 1:
                                    mask_wps_unpruned_nonexistent[idx_state] &= not env.ijxd2state(i_state + 1, j_state, x_state) in env.DP_info["states_reachable"]
                                if j_state > 0:
                                    mask_wps_unpruned_nonexistent[idx_state] &= not env.ijxd2state(i_state, j_state - 1, x_state) in env.DP_info["states_reachable"]
                                if j_state < env.height - 1:
                                    mask_wps_unpruned_nonexistent[idx_state] &= not env.ijxd2state(i_state, j_state + 1, x_state) in env.DP_info["states_reachable"]
                            elif env.name_game == "RandDistShift":
                                if i_state > 0:
                                    mask_wps_unpruned_nonexistent[idx_state] &= not env.ijd2state(i_state - 1, j_state) in env.DP_info["states_reachable"]
                                if i_state < env.width - 1:
                                    mask_wps_unpruned_nonexistent[idx_state] &= not env.ijd2state(i_state + 1, j_state) in env.DP_info["states_reachable"]
                                if j_state > 0:
                                    mask_wps_unpruned_nonexistent[idx_state] &= not env.ijd2state(i_state, j_state - 1) in env.DP_info["states_reachable"]
                                if j_state < env.height - 1:
                                    mask_wps_unpruned_nonexistent[idx_state] &= not env.ijd2state(i_state, j_state + 1) in env.DP_info["states_reachable"]
                        if env.name_game == "SwordShieldMonster" and state_reachable_frominit and idx_state:
                            x_targ = int(vertices_unpruned["ijxds"][idx_state][2])
                            targ_irreversible = False
                            if x_curr == 1 and (x_targ == 0 or x_targ == 2):
                                targ_irreversible = True
                            elif x_curr == 2 and (x_targ == 0 or x_targ == 1):
                                targ_irreversible = True
                            elif x_curr == 3 and x_targ < 3:
                                targ_irreversible = True
                            mask_wps_unpruned_irreversible[idx_state] = targ_irreversible
                    mask_wps_generated_unpruned_nonexistent = mask_wps_unpruned_nonexistent[1:-1]
                    num_wps_generated_unpruned_nonexistent = int(mask_wps_generated_unpruned_nonexistent.sum())
                    writer.add_scalar(f"{prefix_plan}/num_wps_generated_unpruned_nonexistent", num_wps_generated_unpruned_nonexistent, step_record)
                    writer.add_scalar(f"{prefix_plan}/ratio_wps_generated_unpruned_nonexistent", num_wps_generated_unpruned_nonexistent / (self.num_waypoints_unpruned - 2), step_record)
                    if env.name_game == "SwordShieldMonster" and not mask_wps_generated_unpruned_nonexistent.all():
                        num_wps_generated_unpruned_irreversible = int(mask_wps_unpruned_irreversible.sum())
                        writer.add_scalar(f"{prefix_plan}/num_wps_generated_unpruned_irreversible", num_wps_generated_unpruned_irreversible, step_record)
                        writer.add_scalar(f"{prefix_plan}/ratio_wps_generated_unpruned_irreversible", num_wps_generated_unpruned_irreversible / (self.num_waypoints_unpruned - 2), step_record)
                        writer.add_scalar(f"{prefix_plan}/ratio_wps_generated_unpruned_irreversible_existent", num_wps_generated_unpruned_irreversible / (self.num_waypoints_unpruned - 2 - num_wps_generated_unpruned_nonexistent), step_record)
                        vertices_unpruned.update(mask_irreversible=mask_wps_unpruned_irreversible)
                    vertices_unpruned.update(mask_nonexistent=mask_wps_unpruned_nonexistent)

                if self.unique_obses:
                    indices_unique_obses = find_unique(vertices_unpruned["obses"][:, :, :, 0], must_keep=[0, -1])
                if self.unique_codes:
                    indices_unique_codes = find_unique(vertices_unpruned["codes"].flatten(1), must_keep=[0, -1])
                if self.unique_obses and self.unique_codes:
                    indices_unique = np.intersect1d(indices_unique_obses, indices_unique_codes).tolist()
                elif self.unique_obses and not self.unique_codes:
                    indices_unique = indices_unique_obses
                elif not self.unique_obses and self.unique_codes:
                    indices_unique = indices_unique_codes
                else:
                    indices_unique = None

                if debug:
                    if self.unique_obses:
                        writer.add_scalar(f"{prefix_plan}/num_waypoints_unpruned_unique_obs", len(indices_unique_obses), step_record)
                    if self.unique_codes:
                        writer.add_scalar(f"{prefix_plan}/num_waypoints_unpruned_unique_code", len(indices_unique_codes), step_record)
                    writer.add_scalar(f"{prefix_plan}/num_waypoints_unpruned_unique", len(indices_unique), step_record)

                if indices_unique is None:
                    vertices_pruned = vertices_unpruned.copy()
                else:
                    assert indices_unique[0] == 0 and indices_unique[-1] == self.num_waypoints_unpruned - 1
                    vertices_pruned = {}
                    if "ijxds" in vertices_unpruned:
                        vertices_pruned["ijxds"] = vertices_unpruned["ijxds"][indices_unique, :]
                    if "states" in vertices_unpruned:
                        vertices_pruned["states"] = vertices_unpruned["states"][indices_unique]
                    if "mask_nonexistent" in vertices_unpruned:
                        vertices_pruned["mask_nonexistent"] = vertices_unpruned["mask_nonexistent"][indices_unique]
                    if "mask_irreversible" in vertices_unpruned:
                        vertices_pruned["mask_irreversible"] = vertices_unpruned["mask_irreversible"][indices_unique]
                    vertices_pruned["obses"] = vertices_unpruned["obses"][indices_unique]
                    vertices_pruned["codes"] = vertices_unpruned["codes"][indices_unique]

                self.vertices_unpruned = vertices_unpruned
                edges_pruned = self.get_abstract_graph(vertices_pruned, env=env, save_wp_existing_obses=True)
                edges_pruned["omegas"][0] = False
                edges_pruned["discounts"][edges_pruned["omegas"]] = 0.0
                edges_pruned["rewards"][edges_pruned["omegas"]] = 0.0

                if indices_unique is None or len(indices_unique) > self.num_waypoints:
                    dist = edges_pruned["distances"].clone()
                    dist[edges_pruned["omegas"]] = float('inf')
                    mask_reachable_from_curr = reachability_from_distances(dist, idx_start=0, dist_cutoff=self.dist_cutoff)
                    mask_reachable_from_curr[-1] = True
                    num_waypoints_chosen = int(mask_reachable_from_curr.sum())
                    if not mask_reachable_from_curr.all():
                        vertices_pruned, edges_pruned = self.prune_proxy_with_indices(vertices=vertices_pruned, edges=edges_pruned, indices=mask_reachable_from_curr)
                        self.obses_wps_existing = self.obses_wps_existing[mask_reachable_from_curr]
                    if debug:
                        writer.add_scalar(f"{prefix_plan}/ratio_reachable_waypoints", float(num_waypoints_chosen / mask_reachable_from_curr.shape[0]), step_record)
                        writer.add_scalar(f"{prefix_plan}/num_reachable_waypoints", num_waypoints_chosen, step_record)
                if edges_pruned["distances"].shape[0] > self.num_waypoints:
                    dist = edges_pruned["distances"].clamp_(0, 1000)
                    dist[edges_pruned["omegas"]] = 1000
                    dist = torch.minimum(dist, dist.T)
                    indices_chosen, _, _ = k_medoids(dist, self.num_waypoints, [0, dist.shape[0] - 1])
                    assert indices_chosen[0] == 0 and indices_chosen[-1] == dist.shape[0] - 1
                    num_waypoints_chosen = len(indices_chosen)
                    vertices_pruned, edges_pruned = self.prune_proxy_with_indices(vertices=vertices_pruned, edges=edges_pruned, indices=indices_chosen)
                    self.obses_wps_existing = self.obses_wps_existing[indices_chosen]
                self.proxy_graph_curr = edges_pruned | vertices_pruned
                self.obses_wps_existing = self.obses_wps_existing[1:]
                assert self.obses_wps_existing.shape[0] <= self.num_waypoints - 1
                if self.cvae is None:
                    self.waypoints_existing = vertices_pruned["ijxds"][1:]
                else:
                    self.waypoints_existing = self.encoder_wp(self.obses_wps_existing).reshape(self.obses_wps_existing.shape[0], -1)
                self.proxy_graph_curr["selected"] = np.zeros(self.proxy_graph_curr["omegas"].shape[0])
                self.waypoint_last_reached = None
            else:
                if self.waypoint_curr is None:
                    if self.cvae is None:
                        self.waypoint_curr = self.encoder_wp(obs_curr, env)
                    else:
                        self.waypoint_curr = self.encoder_wp(self.obs2tensor(obs_curr))
                if self.cvae is None:
                    self.waypoints_existing = self.proxy_graph_curr["ijxds"][1:]
                else:
                    self.waypoints_existing = self.encoder_wp(self.obses_wps_existing).reshape(self.obses_wps_existing.shape[0], -1)
                vertices_existing = {"obses": self.obses_wps_existing, "codes": self.waypoints_existing}
                if "ijxds" in self.proxy_graph_curr:
                    vertices_existing["ijxds"] = self.proxy_graph_curr["ijxds"][1:]
                if "states" in self.proxy_graph_curr:
                    vertices_existing["states"] = self.proxy_graph_curr["states"][1:]
                if "mask_nonexistent" in self.proxy_graph_curr:
                    vertices_existing["mask_nonexistent"] = self.proxy_graph_curr["mask_nonexistent"][1:]
                if "mask_irreversible" in self.proxy_graph_curr:
                    vertices_existing["mask_irreversible"] = self.proxy_graph_curr["mask_irreversible"][1:]
                self.proxy_graph_curr.update(self.get_abstract_graph(vertices_existing, obs_curr=obs_curr, env=env, save_wp_existing_obses=False))
            assert self.proxy_graph_curr["distances"].shape[0] <= self.num_waypoints

            omegas_plan = self.proxy_graph_curr["omegas"]
            discounts_plan = self.proxy_graph_curr["discounts"].clone()
            rewards_plan = self.proxy_graph_curr["rewards"].clone()
            distances_plan = self.proxy_graph_curr["distances"].clone()

            mask_cutoff = self.proxy_graph_curr["distances"] > self.dist_cutoff
            mask_cutoff.fill_diagonal_(True)
            if not self.optimal_plan and mask_cutoff[0, :].all():
                return self.get_random_action()
            mask_cutoff[omegas_plan] = True
            discounts_plan.masked_fill_(mask_cutoff, 0.0)
            rewards_plan.masked_fill_(mask_cutoff, 0.0)
            distances_plan.masked_fill_(mask_cutoff, 1024.0)

            Q, num_iters_plan, converged = abstract_planning(discounts_plan, rewards_plan, max_iters=5, no_loop=True)

            if self.optimal_policy:
                self.Q_oracle, self.ijxd_targ_oracle = None, None

            if debug_visualize and generate_graph:
                if self.cvae is not None:
                    if obs_curr_tensor is None:
                        obs_curr_tensor = self.obs2tensor(obs_curr)
                self.visualize_proxy(ijxds_highlight=vertices_unpruned["ijxds"], wp_graph=dict(ijxds=self.proxy_graph_curr["ijxds"], distances=self.proxy_graph_curr["distances"], rewards=self.proxy_graph_curr["rewards"], discounts=self.proxy_graph_curr["discounts"], omegas=self.proxy_graph_curr["omegas"]), env=env, obs_curr=obs_curr, writer=writer, step_record=step_record, prefix_vis=prefix_vis)
                self.visualize_proxy(ijxds_highlight=self.proxy_graph_curr["ijxds"], wp_graph=dict(ijxds=self.proxy_graph_curr["ijxds"], Q=Q, distances=distances_plan, rewards=rewards_plan, discounts=discounts_plan, omegas=omegas_plan), env=env, obs_curr=obs_curr, writer=writer, step_record=step_record, prefix_vis=prefix_vis, suffix_vis="_plan")
                
            if debug:
                writer.add_scalar(f"{prefix_plan}/num_iters", int(num_iters_plan), step_record)
                writer.add_scalar(f"{prefix_plan}/VI_converged", float(converged), step_record)
                if converged:
                    num_iters_plan_converge = num_iters_plan
                else:
                    _, num_iters_plan_converge, _ = abstract_planning(discounts_plan, rewards_plan, omegas_plan, max_iters=1000)
                writer.add_scalar(f"{prefix_plan}/num_iters_converge", int(num_iters_plan_converge), step_record)

            Q_wp_curr = Q[0].cpu().numpy()
            Q_wp_curr[0] = -float('inf')
            idx_targs = np.where(np.abs(np.max(Q_wp_curr) - Q_wp_curr) < 1e-5)[0].tolist()
            if len(idx_targs) > 1:
                distances_targs = np.take_along_axis(distances_plan[0, :].cpu().numpy(), np.array(idx_targs), -1)
                idx_targs = [idx_targs[index] for index in distances_targs.argsort().tolist()]
                assert len(idx_targs), f"distances_targs.argsort().tolist(): {distances_targs.argsort().tolist()}"
            if discounts_plan.shape[0] == 2:
                idx_targ = 1
            else:
                try:
                    idx_targ = int(idx_targs[0])
                except:
                    print("error in idx_targs:", idx_targs)
                    raise RuntimeError("what happened")

            if self.optimal_plan or debug:
                if env.name_game == "RandDistShift":
                    ijxd_curr = np.array(env.obs2ijd(obs_curr)[: 3 - int(env.ignore_dir)])
                elif env.name_game == "SwordShieldMonster":
                    ijxd_curr = np.array(env.obs2ijxd(obs_curr)[: 4 - int(env.ignore_dir)])
                proxy_graph_GT = {"ijxds": np.concatenate([ijxd_curr.reshape(1, *self.proxy_graph_curr["ijxds"].shape[1:]), self.proxy_graph_curr["ijxds"][1:]], 0)}
                if env.name_game == "RandDistShift":
                    proxy_graph_GT["states"] = np.array([env.ijd2state(*ijxd_curr)] + self.proxy_graph_curr["states"][1:].tolist())
                elif env.name_game == "SwordShieldMonster":
                    proxy_graph_GT["states"] = np.array([env.ijxd2state(*ijxd_curr)] + self.proxy_graph_curr["states"][1:].tolist())
                temp = append_GT_graph(env, proxy_graph_GT)
                discounts_GT, distances_GT, rewards_GT, omegas_GT = torch.tensor(temp["discount"], device=discounts_plan.device), torch.tensor(temp["distance"], device=discounts_plan.device), torch.tensor(temp["reward"], device=discounts_plan.device), torch.tensor(temp["done"], device=discounts_plan.device)

                Q_GT, _, _ = abstract_planning(discounts_GT, rewards_GT, omegas_GT, max_iters=5)
                Q_wp_curr_GT = Q_GT[0].cpu().numpy()
                idx_targs_optimal = np.where(np.abs(np.max(Q_wp_curr_GT) - Q_wp_curr_GT) < 1e-5)[0].tolist()
                discounts_targs_optimal = np.take_along_axis(discounts_GT[0, :].cpu().numpy(), np.array(idx_targs_optimal), -1)
                idx_targs_optimal = [idx_targs_optimal[index] for index in (-discounts_targs_optimal).argsort().tolist()]
                if 0 in idx_targs_optimal:
                    idx_targs_optimal = np.setdiff1d(idx_targs_optimal, [0]).tolist()

                if debug:
                    if self.waypoint_curr is None:
                        if self.cvae is None:
                            self.waypoint_curr = self.encoder_wp(obs_curr, env)
                        else:
                            self.waypoint_curr = self.encoder_wp(self.obs2tensor(obs_curr))
                    dist2targ = np.abs(self.waypoint_curr - self.waypoints_existing[idx_targ - 1]).sum()
                    writer.add_scalar(f"{prefix_plan}/dist2targ", dist2targ, step_record)
                    writer.add_scalar(
                        f"{prefix_plan}/dist2targ_robust",
                        np.abs(self.waypoint_curr - self.waypoints_existing[idx_targs_optimal[0] - 1]).sum(),
                        step_record,
                    )
                    writer.add_scalar(f"{prefix_plan}/deviation_Q_optimal", np.abs(Q_wp_curr_GT[idx_targs_optimal[0]] - Q_wp_curr[idx_targs[0]]), step_record)
                    writer.add_scalar(f"{prefix_plan}/deviation_Q_robust", np.abs(Q_wp_curr[idx_targs_optimal[0]] - Q_wp_curr[idx_targs[0]]), step_record)
                    if len(idx_targs_optimal):
                        plan_optimal = float(int(idx_targ) in idx_targs_optimal)
                        writer.add_scalar(f"{prefix_plan}/optimality", plan_optimal, step_record)
                        if len(idx_targs_optimal) > 1:
                            plan_optimal_robust = float(idx_targ == idx_targs_optimal[0])
                            writer.add_scalar(f"{prefix_plan}/optimality_robust", plan_optimal_robust, step_record)
                        mask_targs = np.zeros(proxy_graph_GT["ijxds"].shape[0], dtype=bool)
                        mask_targs[idx_targs] = True
                        mask_targs_optimal = np.zeros(proxy_graph_GT["ijxds"].shape[0], dtype=bool)
                        mask_targs_optimal[idx_targs_optimal] = True
                        writer.add_scalar(f"{prefix_plan}/optimal_intersect", (mask_targs == mask_targs_optimal).sum() / self.num_waypoints, step_record)

                    mask_interest = torch.logical_not(mask_cutoff)
                    mask_interest[:, 0] = False
                    mask_interest[omegas_GT] = False
                    mask_interest *= ~torch.eye(omegas_GT.shape[0], dtype=torch.bool, device=discounts_plan.device)

                    if "mask_nonexistent" in vertices_pruned:
                        if vertices_pruned["mask_nonexistent"].any():
                            mask_nonexistent = torch.tensor(vertices_pruned["mask_nonexistent"], device=discounts_plan.device)

                            mask2d_nonexistent_sources = torch.zeros_like(mask_interest, dtype=torch.bool)
                            mask2d_nonexistent_sources[mask_nonexistent, :] = True
                            mask2d_nonexistent_sources[:, mask_nonexistent] = False
                            mask2d_nonexistent_sources[omegas_GT] = False

                            mask2d_nonexistent_targets = torch.zeros_like(mask_interest, dtype=torch.bool)
                            mask2d_nonexistent_targets[:, mask_nonexistent] = True
                            mask2d_nonexistent_targets[mask_nonexistent, :] = False
                            mask2d_nonexistent_targets[omegas_GT] = False

                            mask2d_nonexistent = mask2d_nonexistent_sources | mask2d_nonexistent_targets

                            mask2d_nonexistent_sources_only = mask2d_nonexistent_sources & ~mask2d_nonexistent_targets
                            mask2d_nonexistent_targets_only = mask2d_nonexistent_targets & ~mask2d_nonexistent_sources
                        else:
                            mask2d_nonexistent, mask2d_nonexistent_sources, mask2d_nonexistent_targets, mask2d_nonexistent_sources_only, mask2d_nonexistent_targets_only = None, None, None, None, None
                    if "mask_irreversible" in vertices_pruned:
                        mask2d_irreversible = torch.zeros_like(mask_interest, dtype=torch.bool)
                        for i in range(0, mask_interest.shape[0]):
                            if omegas_GT[i]:
                                continue
                            for j in range(0, mask_interest.shape[1]):
                                if (mask2d_nonexistent is not None and mask2d_nonexistent[i, j]) or i == j:
                                    continue
                                x_curr, x_targ = int(proxy_graph_GT["ijxds"][i][2]), int(proxy_graph_GT["ijxds"][j][2])
                                if x_curr == 1 and (x_targ == 0 or x_targ == 2):
                                    mask2d_irreversible[i, j] = True
                                elif x_curr == 2 and (x_targ == 0 or x_targ == 1):
                                    mask2d_irreversible[i, j] = True
                                elif x_curr == 3 and x_targ < 3:
                                    mask2d_irreversible[i, j] = True
                    else:
                        mask2d_irreversible = None

                    diff_distances = (distances_GT.clamp(0, self.network_policy.estimator_discount.atoms) - self.proxy_graph_curr["distances"].clamp(0, self.network_policy.estimator_discount.atoms)).abs_()
                    diff_discounts = (discounts_GT - self.proxy_graph_curr["discounts"]).abs_()
                    diff_rewards = (rewards_GT - self.proxy_graph_curr["rewards"]).abs_()

                    if mask_interest.any():
                        deviation_Q = (Q_GT - Q).abs_()[mask_interest].mean().item()
                        writer.add_scalar(f"{prefix_plan}/deviation_Q", deviation_Q, step_record)
                        writer.add_scalar(f"{prefix_debug}/diff_distances", diff_distances[mask_interest].mean().item(), step_record)
                        writer.add_scalar(f"{prefix_debug}/diff_discounts", diff_discounts[mask_interest].mean().item(), step_record)
                        writer.add_scalar(f"{prefix_debug}/diff_rewards", diff_rewards[mask_interest].mean().item(), step_record)

                        mask_zero_discounts = discounts_GT == 0
                        mask_trivial_discounts = mask_zero_discounts * mask_interest
                        if mask_trivial_discounts.any():
                            diff_discounts_trivial = diff_discounts[mask_trivial_discounts]
                            writer.add_scalar(f"{prefix_debug}/diff_discounts_trivial", diff_discounts_trivial.mean().item(), step_record)

                        mask_nontrivial_discounts = ~mask_zero_discounts * mask_interest
                        if mask_nontrivial_discounts.any():
                            writer.add_scalar(f"{prefix_debug}/diff_discounts_nontrivial", diff_discounts[mask_nontrivial_discounts].mean().item(), step_record)
                            writer.add_scalar(f"{prefix_debug}/diff_distances_nontrivial", diff_distances[mask_nontrivial_discounts].mean().item(), step_record)

                        mask_zero_rewards = rewards_GT == 0
                        mask_trivial_rewards = mask_zero_rewards * mask_interest
                        if mask_trivial_rewards.any():
                            writer.add_scalar(f"{prefix_debug}/diff_rewards_trivial", diff_rewards[mask_trivial_rewards].mean().item(), step_record)
                        mask_nontrivial_rewards = ~mask_zero_rewards * mask_interest
                        if mask_nontrivial_rewards.any():
                            writer.add_scalar(f"{prefix_debug}/diff_rewards_nontrivial", diff_rewards[mask_nontrivial_rewards].mean().item(), step_record)

                    if mask2d_nonexistent is not None and mask2d_nonexistent.any():
                        writer.add_scalar(f"{prefix_debug}/diff_distances_nonexistent", diff_distances[mask2d_nonexistent].mean().item(), step_record)
                        writer.add_scalar(f"{prefix_debug}/diff_discounts_nonexistent", diff_discounts[mask2d_nonexistent].mean().item(), step_record)
                        writer.add_scalar(f"{prefix_debug}/diff_rewards_nonexistent", diff_rewards[mask2d_nonexistent].mean().item(), step_record)
                    
                    if mask2d_nonexistent_sources is not None and mask2d_nonexistent_sources.any():
                        writer.add_scalar(f"{prefix_debug}/diff_distances_nonexistent_sources", diff_distances[mask2d_nonexistent_sources].mean().item(), step_record)
                        writer.add_scalar(f"{prefix_debug}/diff_discounts_nonexistent_sources", diff_discounts[mask2d_nonexistent_sources].mean().item(), step_record)
                        writer.add_scalar(f"{prefix_debug}/diff_rewards_nonexistent_sources", diff_rewards[mask2d_nonexistent_sources].mean().item(), step_record)
                    
                    if mask2d_nonexistent_targets is not None and mask2d_nonexistent_targets.any():
                        writer.add_scalar(f"{prefix_debug}/diff_distances_nonexistent_targets", diff_distances[mask2d_nonexistent_targets].mean().item(), step_record)
                        writer.add_scalar(f"{prefix_debug}/diff_discounts_nonexistent_targets", diff_discounts[mask2d_nonexistent_targets].mean().item(), step_record)
                        writer.add_scalar(f"{prefix_debug}/diff_rewards_nonexistent_targets", diff_rewards[mask2d_nonexistent_targets].mean().item(), step_record)

                    if mask2d_nonexistent_sources_only is not None and mask2d_nonexistent_sources_only.any():
                        writer.add_scalar(f"{prefix_debug}/diff_distances_nonexistent_sources_only", diff_distances[mask2d_nonexistent_sources_only].mean().item(), step_record)
                        writer.add_scalar(f"{prefix_debug}/diff_discounts_nonexistent_sources_only", diff_discounts[mask2d_nonexistent_sources_only].mean().item(), step_record)
                        writer.add_scalar(f"{prefix_debug}/diff_rewards_nonexistent_sources_only", diff_rewards[mask2d_nonexistent_sources_only].mean().item(), step_record)
                    
                    if mask2d_nonexistent_targets_only is not None and mask2d_nonexistent_targets_only.any():
                        writer.add_scalar(f"{prefix_debug}/diff_distances_nonexistent_targets_only", diff_distances[mask2d_nonexistent_targets_only].mean().item(), step_record)
                        writer.add_scalar(f"{prefix_debug}/diff_discounts_nonexistent_targets_only", diff_discounts[mask2d_nonexistent_targets_only].mean().item(), step_record)
                        writer.add_scalar(f"{prefix_debug}/diff_rewards_nonexistent_targets_only", diff_rewards[mask2d_nonexistent_targets_only].mean().item(), step_record)

                    if mask2d_irreversible is not None and mask2d_irreversible.any():
                        writer.add_scalar(f"{prefix_debug}/diff_distances_irreversible", diff_distances[mask2d_irreversible].mean().item(), step_record)
                        writer.add_scalar(f"{prefix_debug}/diff_discounts_irreversible", diff_discounts[mask2d_irreversible].mean().item(), step_record)
                        writer.add_scalar(f"{prefix_debug}/diff_rewards_irreversible", diff_rewards[mask2d_irreversible].mean().item(), step_record)

                    writer.add_scalar(f"{prefix_debug}/diff_omegas", (omegas_GT != omegas_plan).float().mean().item(), step_record)
                if self.optimal_plan:
                    idx_targ = int(idx_targs_optimal[0])
            assert idx_targ > 0, f"self-loop planned at step {self.steps_interact:d}: {self.waypoint_curr.tolist()}"
            if self.always_select_goal:
                idx_targ = len(self.waypoints_existing)
            self.idx_waypoint_targ = idx_targ - 1
            self.waypoint_targ = self.waypoints_existing[idx_targ - 1]
            if "ijxds" in self.proxy_graph_curr:
                self.ijxd_targ = self.proxy_graph_curr["ijxds"][idx_targ]
            self.proxy_graph_curr["selected"][self.idx_waypoint_targ] = True
            self.obs_wp_targ = None if self.cvae is None else self.obses_wps_existing[[idx_targ - 1]]
            if self.optimal_policy:
                if self.ijxd_targ_oracle is None or (np.array(self.ijxd_targ_oracle) != self.ijxd_targ).any():
                    ret = env.generate_oracle(ijxd_targ=self.proxy_graph_curr["ijxds"][idx_targ].tolist())
                    self.Q_oracle, self.ijxd_targ_oracle = ret["Q_optimal"], ret["ijxd_targ"]
            if debug and "mask_nonexistent" in vertices_pruned:
                writer.add_scalar(f"{prefix_plan}/targ_nonexistent", float(vertices_pruned["mask_nonexistent"][idx_targ]), step_record)
            if debug and "mask_irreversible" in vertices_pruned:
                writer.add_scalar(f"{prefix_plan}/targ_irreversible", float(vertices_pruned["mask_irreversible"][idx_targ]), step_record)
        if self.optimal_policy:
            assert self.Q_oracle is not None and self.ijxd_targ_oracle is not None
            q = self.Q_oracle[env.obs2state(obs_curr)]
            if (q == 0).all():
                return self.get_random_action()
            else:
                return q.argmax()
        if obs_curr_tensor is None:
            obs_curr_tensor = self.obs2tensor(obs_curr)
        if self.cvae is None:
            return self.Q_conditioned(obs_curr_tensor, waypoint_targ=self.waypoint_targ, obs_targ=None, type_curr="obs", env=env).argmax().item()
        else:
            return self.Q_conditioned(obs_curr_tensor, waypoint_targ=None, obs_targ=self.obs_wp_targ, type_curr="obs", env=env).argmax().item()

    def step(self, obs_curr, action, reward, obs_next, done, writer=None, add_to_buffer=True, increment_steps=True, idx_env=None):
        if increment_steps:
            self.steps_interact += 1
        if add_to_buffer and obs_next is not None:
            sample = {"obs": np.array(obs_curr), "act": action, "rew": reward, "done": done, "next_obs": np.array(obs_next)}
            if idx_env is not None:
                sample["idx_env"] = idx_env
            self.add_to_buffer(sample)


class DQN_SKIPPER(DQN_SKIPPER_BASE):
    def __init__(
        self,
        env,
        network_policy,
        network_target=None,
        freq_plan=4,
        num_waypoints=16,
        waypoint_strategy="once",
        always_select_goal=False,
        optimal_plan=False,
        optimal_policy=False,
        dist_cutoff=8,
        gamma=0.99,
        gamma_int=0.95,
        type_intrinsic_reward="sparse",
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
        valid_waypoints_only=False,
        no_lava_waypoints=False,
        hrb=None,
        silent=False,
        transform_discount_target=True,
        num_waypoints_unpruned=32,
        prob_relabel_generateJIT=0.0,
        no_Q_head=False,
        unique_codes=False,
        unique_obses=True,
        nonsingleton=False,
    ):
        super(DQN_SKIPPER, self).__init__(
            env,
            network_policy,
            freq_plan=freq_plan,
            num_waypoints=num_waypoints,
            waypoint_strategy=waypoint_strategy,
            always_select_goal=always_select_goal,
            optimal_plan=optimal_plan,
            optimal_policy=optimal_policy,
            dist_cutoff=dist_cutoff,
            gamma=gamma,
            gamma_int=gamma_int,
            type_intrinsic_reward=type_intrinsic_reward,
            clip_reward=clip_reward,
            exploration_fraction=exploration_fraction,
            epsilon_final_train=epsilon_final_train,
            epsilon_eval=epsilon_eval,
            steps_total=steps_total,
            prioritized_replay=prioritized_replay,
            func_obs2tensor=func_obs2tensor,
            device=device,
            seed=seed,
            valid_waypoints_only=valid_waypoints_only,
            no_lava_waypoints=no_lava_waypoints,
            hrb=hrb,
            silent=silent,
            transform_discount_target=transform_discount_target,
            num_waypoints_unpruned=num_waypoints_unpruned,
            prob_relabel_generateJIT=prob_relabel_generateJIT,
            no_Q_head=no_Q_head,
            unique_codes=unique_codes,
            unique_obses=unique_obses,
            nonsingleton=nonsingleton,
        )

        self.optimizer = eval("torch.optim.%s" % type_optimizer)(self.network_policy.parameters(), lr=lr, eps=eps)
        if network_target is None:
            self.network_target = copy.deepcopy(self.network_policy)
        else:
            self.network_target = network_target
        if self.network_target.cvae is not None:
            self.network_target.cvae.to("cpu")
            self.network_target.cvae = None
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

        generate_new_targ2 = False
        if self.prob_relabel_generateJIT > 0:
            if np.random.rand() < float(self.prob_relabel_generateJIT):
                generate_new_targ2 = True            

        if generate_new_targ2:
            with torch.no_grad():
                (
                    priorities_original,
                    loss_TD,
                    loss_discount,
                    loss_reward,
                    loss_omega,
                    omega_logits_pred,
                    batch_state_curr,
                    state_local_curr,
                    state_local_next,
                    state_local_next_targetnet
                ) = self.calculate_multihead_error(batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, batch_obs_targ2, batch_reward_int=None)

                if self.cvae is None:
                    batch_obs_targ2 = torch.zeros_like(batch_obs_targ)
                    for i in range(self.size_batch):
                        env = copy.deepcopy(self.env)
                        obs_curr = batch_obs_curr[i].cpu().numpy()
                        env.load_layout_from_obs(obs_curr)
                        targ = generate_random_waypoints(
                            env,
                            1,
                            include_goal=False,
                            include_agent=False,
                            generate_DP_info=False,
                            render=False,
                            valid_only=False,
                            no_lava=True,
                            return_dist=False,
                            return_obs=True,
                            unique=False,
                            obs_curr=batch_obs_curr[i].cpu().numpy(),
                        )
                        batch_obs_targ2[i] = self.obs2tensor(targ["obses"])
                        del env, obs_curr, targ
                else:
                    batch_obs_targ2 = self.cvae.imagine_batch_from_obs(batch_obs_curr)
        else:
            priorities_original = None
        (
            priorities,
            loss_TD,
            loss_discount,
            loss_reward,
            loss_omega,
            omega_logits_pred,
            batch_state_curr,
            state_local_curr,
            state_local_next,
            state_local_next_targetnet
        ) = self.calculate_multihead_error(batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, batch_obs_targ2, batch_reward_int=None)
        if priorities_original is not None:
            priorities = priorities_original
        if self.cvae is not None:
            (
                loss_cvae,
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
            ) = self.cvae.compute_loss(batch_processed, debug=debug)

        loss_overall = loss_TD + loss_discount + loss_reward + loss_omega
        if self.prioritized_replay:
            assert weights is not None
            error_overall_weighted = (loss_overall * weights.detach()).mean()

        else:
            error_overall_weighted = loss_overall.mean()

        if self.cvae is not None:
            error_overall_weighted += loss_cvae.mean()

        self.optimizer.zero_grad(set_to_none=True)
        error_overall_weighted.backward()

        if debug:
            with torch.no_grad():
                grads = [param.grad.detach().flatten() for param in self.network_policy.parameters() if param.grad is not None]
                norm_grad = torch.cat(grads).norm().item()
        torch.nn.utils.clip_grad_value_(self.network_policy.parameters(), 1.0)
        self.optimizer.step()

        with torch.no_grad():
            if self.prioritized_replay and not generate_new_targ2:
                self.hrb.update_priorities(batch_idxes, priorities.detach().cpu().numpy())
                if debug:
                    writer.add_scalar("Train/priorities", priorities.mean().item(), self.steps_processed)

            if debug:
                if self.cvae is not None:
                    writer.add_scalar("Train_CVAE/loss_overall", loss_cvae.mean().item(), self.steps_processed)
                    writer.add_scalar("Train_CVAE/loss_recon", loss_recon.mean().item(), self.steps_processed)
                    writer.add_scalar("Train_CVAE/loss_entropy", loss_entropy.mean().item(), self.steps_processed)
                    if loss_align is not None:
                        writer.add_scalar("Train_CVAE/loss_align", loss_align.mean().item(), self.steps_processed)
                    writer.add_scalar("Train_CVAE/dist_L1", dist_L1_mean.item(), self.steps_processed)
                    writer.add_scalar("Train_CVAE/dist_L1_nontrivial", dist_L1_nontrivial.item(), self.steps_processed)
                    writer.add_scalar("Train_CVAE/dist_L1_trivial", dist_L1_trivial.item(), self.steps_processed)
                    writer.add_scalar("Train_CVAE/uniformity", uniformity.item(), self.steps_processed)
                    writer.add_scalar("Train_CVAE/ratio_imperfect_recon", 1.0 - ratio_perfect_recon.item(), self.steps_processed)
                    writer.add_scalar("Train_CVAE/ratio_aligned", ratio_aligned.item(), self.steps_processed)

                writer.add_scalar("Debug/norm_rep", torch.sqrt((batch_state_curr.flatten(1) ** 2).sum(-1)).mean().item(), self.steps_processed)
                writer.add_scalar("Debug/norm_rep_local", torch.sqrt((state_local_curr**2).sum(-1)).mean().item(), self.steps_processed)
                writer.add_scalar("Debug/norm_grad", norm_grad, self.steps_processed)
                writer.add_scalar("Train/loss_TD", loss_TD.mean().item(), self.steps_processed)
                writer.add_scalar("Train/loss_discount", loss_discount.mean().item(), self.steps_processed)
                writer.add_scalar("Train/loss_reward", loss_reward.mean().item(), self.steps_processed)
                writer.add_scalar("Train/loss_omega", loss_omega.mean().item(), self.steps_processed)
                if omega_logits_pred is not None:
                    omega_pred = omega_logits_pred.argmax(-1).bool()
                    acc_omega = (batch_done == omega_pred).sum() / batch_done.shape[0]
                    writer.add_scalar("Debug/acc_omega", acc_omega.item(), self.steps_processed)

    def sync_parameters(self):
        """
        synchronize the parameters of self.network_policy and self.network_target
        this is hard sync, maybe a softer version is going to do better
        cvae not synced, since we don't need it for target network
        """
        self.network_target.encoder.load_state_dict(self.network_policy.encoder.state_dict())
        self.network_target.binder.load_state_dict(self.network_policy.binder.state_dict())
        if self.network_policy.estimator_Q is not None:
            self.network_target.estimator_Q.load_state_dict(self.network_policy.estimator_Q.state_dict())
        self.network_target.estimator_discount.load_state_dict(self.network_policy.estimator_discount.state_dict())
        self.network_target.estimator_reward.load_state_dict(self.network_policy.estimator_reward.state_dict())
        self.network_target.estimator_omega.load_state_dict(self.network_policy.estimator_omega.state_dict())
        if not self.silent:
            print("policy-target parameters synced")


def create_DQN_Skipper_network(args, env, dim_embed, num_actions, device, share_memory=False):
    from models import Encoder_MiniGrid, Binder_MiniGrid, Predictor_MiniGrid

    if args.activation == "relu":
        activation = torch.nn.ReLU
    elif args.activation == "elu":
        activation = torch.nn.ELU
    elif args.activation == "leakyrelu":
        activation = torch.nn.LeakyReLU
    elif args.activation == "silu":
        activation = torch.nn.SiLU

    encoder = Encoder_MiniGrid(dim_embed, obs_sample=env.reset(), norm=bool(args.layernorm), append_pos=bool(args.append_pos), activation=activation)
    encoder.to(device)
    if share_memory:
        encoder.share_memory()

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
    if share_memory:
        binder.share_memory()

    if args.no_Q_head:
        estimator_Q = None
    else:
        if args.type_intrinsic_reward == "sparse":
            dict_head_Q = {"len_predict": num_actions, "dist_out": True, "value_min": 0, "value_max": 1, "atoms": args.atoms_value, "classify": False}
        elif args.type_intrinsic_reward == "dense":
            dict_head_Q = {
                "len_predict": num_actions,
                "dist_out": True,
                "value_min": -float(args.atoms_value),
                "value_max": -1,
                "atoms": args.atoms_value,
                "classify": False,
            }
        else:
            raise NotImplementedError()

        estimator_Q = Predictor_MiniGrid(
            num_actions,
            len_input=binder.len_out,
            depth=args.depth_hidden,
            width=args.width_hidden,
            norm=bool(args.layernorm),
            activation=activation,
            dict_head=dict_head_Q,
        )
        estimator_Q.to(device)
        if share_memory:
            estimator_Q.share_memory()

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
    estimator_discount = Predictor_MiniGrid(
        num_actions,
        len_input=binder.len_out,
        depth=args.depth_hidden,
        width=args.width_hidden,
        norm=bool(args.layernorm),
        activation=activation,
        dict_head=dict_head_discount,
    )
    if args.transform_discount_target:
        estimator_discount.histogram_converter.support_distance = torch.arange(1, args.atoms_discount + 1, device=device, dtype=torch.float32)
        estimator_discount.histogram_converter.support_discount = torch.pow(args.gamma, estimator_discount.histogram_converter.support_distance)
    else:
        estimator_discount.histogram_converter.support_discount = torch.linspace(0, args.gamma, args.atoms_discount, device=device, dtype=torch.float32)
        estimator_discount.histogram_converter.support_distance = torch.log(estimator_discount.histogram_converter.support_discount) / np.log(args.gamma)
        estimator_discount.histogram_converter.support_distance.clamp_(1, 250)
    estimator_discount.histogram_converter.support_override = True
    estimator_discount.to(device)
    if share_memory:
        estimator_discount.share_memory()

    dict_head_reward = {
        "len_predict": num_actions,
        "dist_out": True,
        "value_min": args.value_min,
        "value_max": args.value_max,
        "atoms": args.atoms_reward,
        "classify": False,
    }
    estimator_reward = Predictor_MiniGrid(
        num_actions,
        len_input=binder.len_out,
        depth=args.depth_hidden,
        width=args.width_hidden,
        norm=bool(args.layernorm),
        activation=activation,
        dict_head=dict_head_reward,
    )
    estimator_reward.to(device)
    if share_memory:
        estimator_reward.share_memory()

    dict_head_omega = {"len_predict": 1, "dist_out": True, "value_min": 0.0, "value_max": 1.0, "atoms": 2, "classify": True}
    estimator_omega = Predictor_MiniGrid(
        num_actions,
        len_input=args.len_rep,
        depth=args.depth_hidden,
        width=args.width_hidden,
        norm=bool(args.layernorm),
        activation=activation,
        dict_head=dict_head_omega,
    )
    estimator_omega.to(device)
    if share_memory:
        estimator_omega.share_memory()

    if args.cvae:
        from models import CVAE_MiniGrid
        if "RandDistShift" in args.game:
            from models import Encoder_MiniGrid_RDS, Decoder_MiniGrid_RDS
            encoder_CVAE = Encoder_MiniGrid_RDS()
            decoder_CVAE = Decoder_MiniGrid_RDS()
            num_classes_abstract = 1
            num_categoricals, num_categories = 6, 2
        elif "SwordShieldMonster" in args.game:
            from models import Encoder_MiniGrid_SSM, Decoder_MiniGrid_SSM
            encoder_CVAE = Encoder_MiniGrid_SSM()
            decoder_CVAE = Decoder_MiniGrid_SSM()
            num_classes_abstract = 4
            num_categoricals, num_categories = 6, 2
        else:
            raise NotImplementedError()

        cvae = CVAE_MiniGrid(
            encoder_CVAE,
            decoder_CVAE,
            minigridobs2tensor(env.reset()),
            num_categoricals=num_categoricals,
            num_categories=num_categories,
            activation=activation,
            num_classes_abstract=num_classes_abstract,
        )
        cvae.to(device)
        if share_memory:
            cvae.share_memory()
    else:
        cvae = None

    network_policy = DQN_SKIPPER_NETWORK(encoder, binder, estimator_Q, estimator_discount, estimator_reward, estimator_omega, cvae=cvae)
    if share_memory:
        network_policy.share_memory()
    return network_policy


def create_DQN_Skipper_agent(
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
        network_policy = create_DQN_Skipper_network(args, env, dim_embed, num_actions, device=device, share_memory=False)

    if inference_only:
        agent = DQN_SKIPPER_BASE(
            env,
            network_policy,
            freq_plan=args.freq_plan,
            num_waypoints=args.num_waypoints,
            waypoint_strategy=args.waypoint_strategy,
            always_select_goal=args.always_select_goal,
            optimal_plan=args.optimal_plan,
            optimal_policy=args.optimal_policy,
            gamma=args.gamma,
            gamma_int=args.gamma_int,
            type_intrinsic_reward=args.type_intrinsic_reward,
            steps_total=args.steps_max,
            prioritized_replay=bool(args.prioritized_replay),
            device=device,
            seed=args.seed,
            valid_waypoints_only=args.valid_waypoints_only,
            no_lava_waypoints=args.no_lava_waypoints,
            hrb=hrb,
            silent=silent,
            transform_discount_target=args.transform_discount_target,
            num_waypoints_unpruned=args.num_waypoints_unpruned,
            prob_relabel_generateJIT=args.prob_relabel_generateJIT,
            no_Q_head=args.no_Q_head,
            unique_codes=args.unique_codes,
            unique_obses=args.unique_obses,
            nonsingleton=args.nonsingleton,
        )
    else:
        agent = DQN_SKIPPER(
            env,
            network_policy,
            freq_plan=args.freq_plan,
            num_waypoints=args.num_waypoints,
            waypoint_strategy=args.waypoint_strategy,
            always_select_goal=args.always_select_goal,
            optimal_plan=args.optimal_plan,
            optimal_policy=args.optimal_policy,
            gamma=args.gamma,
            gamma_int=args.gamma_int,
            type_intrinsic_reward=args.type_intrinsic_reward,
            steps_total=args.steps_max,
            prioritized_replay=bool(args.prioritized_replay),
            freq_train=args.freq_train,
            freq_targetsync=args.freq_targetsync,
            lr=args.lr,
            size_batch=args.size_batch,
            device=device,
            seed=args.seed,
            valid_waypoints_only=args.valid_waypoints_only,
            no_lava_waypoints=args.no_lava_waypoints,
            hrb=hrb,
            silent=silent,
            transform_discount_target=args.transform_discount_target,
            num_waypoints_unpruned=args.num_waypoints_unpruned,
            prob_relabel_generateJIT=args.prob_relabel_generateJIT,
            no_Q_head=args.no_Q_head,
            unique_codes=args.unique_codes,
            unique_obses=args.unique_obses,
            nonsingleton=args.nonsingleton,
        )
    return agent
