import numpy as np, random
import torch
import torch.nn as nn

import elements
import common

from dreamerv2 import expl


class Agent(common.Module):

    def __init__(self, config, logger, actspce, step):  # , dataset):
        super(Agent, self).__init__()
        self.config = config
        self._logger = logger  # not used
        self._action_space = actspce
        self._num_act = actspce.n if hasattr(actspce, 'n') else actspce.shape[0]
        self._should_expl = elements.Until(int(
            config.expl_until / config.action_repeat))
        self._counter = step
        self.step = step


        rd_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        with torch.random.fork_rng():
            if config.evaluator:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                from common.evaluator import TargetEvaluator
                self.evaluator = TargetEvaluator(
                    type_action="continuous",
                    len_action=self._num_act,
                    num_actions=None,
                    len_state=int("stoch" in config.evaluator_rep) * config.rssm["stoch"] * config.rssm["discrete"] + int("deter" in config.evaluator_rep) * config.rssm["deter"],
                    len_target=int("stoch" in config.evaluator_rep) * config.rssm["stoch"] * config.rssm["discrete"] + int("deter" in config.evaluator_rep) * config.rssm["deter"],
                    atoms=config.evaluator_atoms,
                    gamma=config.discount,
                    create_targetnet=True,
                    interval_sync_targetnet=config.evaluator_sync_every, # how many loss computations before automatically syncing the targetnet, the default is drawn from DQN
                    create_optimizer=True,
                    type_optimizer="Adam",
                    lr=config.evaluator_lr,
                    encoder_state=None,
                    encoder_target=None,
                )
                self.evaluator.to(device)
                if config.evaluator_h_deter_realism:
                    self.cov_deter_EMA = None
                if config.evaluator_h_value_realism:
                    self.cov_value_EMA = None
                if config.evaluator_h_pcont_realism:
                    self.cov_pcont_EMA = None
                if config.evaluator_h_reward_realism:
                    self.cov_reward_EMA = None
            else:
                self.evaluator = None
        random.setstate(rd_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        

        self.wm = WorldModel(self.step, config)
        self._task_behavior = ActorCritic(config, self.step, self._num_act)
        reward = lambda f, s, a: self.wm.heads['reward'](f).mode
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(actspce),
            plan2explore=lambda: expl.Plan2Explore(
                config, self.wm, self._num_act, self.step, reward),
            model_loss=lambda: expl.ModelLoss(
                config, self.wm, self._num_act, self.step, reward),
        )[config.expl_behavior]()

        # init modules without optimizers (once in opt)
        with torch.no_grad():
            channels = 1 if config.grayscale else 3
            if tuple(config.encoder['keys']) == ('flatten',):
                image = torch.zeros(1, 4, np.prod([channels, *config.image_size]))
            else:
                image = torch.zeros(1, 4, channels, *config.image_size)

            self.train({'image': image,
                        'action': torch.zeros(1, 4, self._num_act),
                        'reward': torch.zeros(1, 4),
                        'discount': torch.zeros(1, 4),
                        'done': torch.zeros(1, 4, dtype=torch.bool)}, update_params=False)

    def policy(self, obs, state=None, mode='train'):
        self.step = self._counter

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=common.ENABLE_FP16):

                if state is None:
                    latent = self.wm.rssm.initial(len(obs['image']), obs['image'].device)
                    action = torch.zeros(len(obs['image']), self._num_act).to(obs['image'].device)
                    state = latent, action
                elif obs['reset'].any():
                    # conversion
                    # state = tf.nest.map_structure(lambda x: x * common.pad_dims(
                    #     1.0 - tf.cast(obs['reset'], x.dtype), len(x.shape)), state)

                    latent, action = state
                    latent = {k: v * common.pad_dims(1.0 - obs['reset'], len(v.shape)) for k, v in latent.items()}
                    action = action * common.pad_dims(1.0 - obs['reset'], len(action.shape))
                    state = latent, action

                latent, action = state
                embed = self.wm.encoder(self.wm.preprocess(obs))
                sample = (mode == 'train') or not self.config.eval_state_mean
                latent, _ = self.wm.rssm.obs_step(latent, action, embed, sample)
                feat = self.wm.rssm.get_feat(latent)
                if mode == 'eval':
                    actor = self._task_behavior.actor(feat)
                    action = actor.mode
                elif self._should_expl(self.step):
                    actor = self._expl_behavior.actor(feat)
                    action = actor.sample()
                else:
                    actor = self._task_behavior.actor(feat)
                    action = actor.sample()
                noise = {'train': self.config.expl_noise, 'eval': self.config.eval_noise}
                action = common.action_noise(action, noise[mode], self._action_space)

                outputs = {'action': action.cpu()}  # no grads for env
                state = (latent, action)

                # no grads for env #FIXME but calling detach() multiple time no prob
                # outputs = {'action': action.detach().cpu()}
                # state = (common.dict_detach(latent), action.detach())
            return outputs, state

    def forward(self, data, state=None):
        return self.train(data, state)

    def train(self, data, state=None, update_params=True, flag_debug=True):
        metrics = {}

        # with torch.no_grad(): #FIXME pudia fazer flag
        state, outputs, mets = self.wm.train(data, state)  # outputs could propagate to behaviour

        histograms_record = {}

        rd_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        with torch.random.fork_rng():
            if self.config.evaluator and update_params:
                size_batch, len_seq = state["stoch"].shape[0], state["stoch"].shape[1]
                with torch.no_grad():
                    stoch_curr = outputs["post"]["stoch"][:, :-1].reshape(size_batch, len_seq - 1, -1).to(self.evaluator.device).reshape(-1, outputs["post"]["stoch"].shape[-2] * outputs["post"]["stoch"].shape[-1])
                    deter_curr = outputs["post"]["deter"][:, :-1].to(self.evaluator.device).reshape(-1, outputs["post"]["deter"].shape[-1])
                    feat_curr = torch.cat([stoch_curr, deter_curr], -1).reshape(-1, stoch_curr.shape[-1] + deter_curr.shape[-1])

                    stoch_next = outputs["post"]["stoch"][:, 1:].reshape(size_batch, len_seq - 1, -1).to(self.evaluator.device).reshape(-1, outputs["post"]["stoch"].shape[-2] * outputs["post"]["stoch"].shape[-1])
                    deter_next = outputs["post"]["deter"][:, 1:].to(self.evaluator.device).reshape(-1, outputs["post"]["deter"].shape[-1])
                    feat_next = torch.cat([stoch_next, deter_next], -1).reshape(-1, stoch_next.shape[-1] + deter_next.shape[-1])

                    action_curr = data["action"][:, :-1].to(self.evaluator.device).reshape(-1, data["action"].shape[-1])
                    mask_done = data["done"].bool()[:, 1:].to(self.evaluator.device).reshape(-1)
                    action_next = self._task_behavior.actor(feat_next).sample()
                    
                    if self.config.evaluator_rep == "stoch":
                        state_rep_curr, state_rep_next = stoch_curr, stoch_next
                    elif self.config.evaluator_rep == "deter":
                        state_rep_curr, state_rep_next = deter_curr, deter_next
                    elif self.config.evaluator_rep == "stoch+deter":
                        state_rep_curr, state_rep_next = feat_curr, feat_next
                    else:
                        raise NotImplementedError("")

                    if self.config.evaluator_h_deter_realism:
                        deter_curr_next = torch.cat([deter_curr, deter_next], -1)
                        cov_deter_curr_next = torch.cov(deter_curr_next.T, correction=1)
                        if self.cov_deter_EMA is None:
                            self.cov_deter_EMA = cov_deter_curr_next
                        else:
                            self.cov_deter_EMA = self.config.evaluator_EMA_decay * cov_deter_curr_next + (1 - self.config.evaluator_EMA_decay) * self.cov_deter_EMA
                        mask_diagonal_cov_deter = torch.eye(self.cov_deter_EMA.shape[0], dtype=torch.bool, device=self.cov_deter_EMA.device)
                        self.cov_deter_EMA[mask_diagonal_cov_deter] = self.cov_deter_EMA[mask_diagonal_cov_deter].clamp_min(1e-8)
                    if self.config.evaluator_h_value_realism:
                        value_curr = self._task_behavior._target_critic(feat_curr).mode.reshape(-1, 1)
                        value_next = self._task_behavior._target_critic(feat_next).mode.reshape(-1, 1)
                        value_curr_next = torch.cat([value_curr, value_next], -1)
                        cov_value_curr_next = torch.cov(value_curr_next.T, correction=1)
                        if self.cov_value_EMA is None:
                            self.cov_value_EMA = cov_value_curr_next
                        else:
                            self.cov_value_EMA = self.config.evaluator_EMA_decay * cov_value_curr_next + (1 - self.config.evaluator_EMA_decay) * self.cov_value_EMA
                        mask_diagonal_cov_value = torch.eye(self.cov_value_EMA.shape[0], dtype=torch.bool, device=self.cov_value_EMA.device)
                        self.cov_value_EMA[mask_diagonal_cov_value] = self.cov_value_EMA[mask_diagonal_cov_value].clamp_min(1e-8)
                    if self.config.evaluator_h_pcont_realism:
                        pcont_curr = self.wm.heads['discount'](feat_curr).mean.reshape(-1, 1)
                        pcont_next = self.wm.heads['discount'](feat_next).mean.reshape(-1, 1)
                        pcont_curr_next = torch.cat([pcont_curr, pcont_next], -1)
                        cov_pcont_curr_next = torch.cov(pcont_curr_next.T, correction=1)
                        if self.cov_pcont_EMA is None:
                            self.cov_pcont_EMA = cov_pcont_curr_next
                        else:
                            self.cov_pcont_EMA = self.config.evaluator_EMA_decay * cov_pcont_curr_next + (1 - self.config.evaluator_EMA_decay) * self.cov_pcont_EMA
                        mask_diagonal_cov_pcont = torch.eye(self.cov_pcont_EMA.shape[0], dtype=torch.bool, device=self.cov_pcont_EMA.device)
                        self.cov_pcont_EMA[mask_diagonal_cov_pcont] = self.cov_pcont_EMA[mask_diagonal_cov_pcont].clamp_min(1e-8)
                    if self.config.evaluator_h_reward_realism:
                        reward_curr = self.wm.heads['reward'](feat_curr).mode.reshape(-1, 1)
                        reward_next = self.wm.heads['reward'](feat_next).mode.reshape(-1, 1)
                        reward_curr_next = torch.cat([reward_curr, reward_next], -1)
                        cov_reward_curr_next = torch.cov(reward_curr_next.T, correction=1)
                        if self.cov_reward_EMA is None:
                            self.cov_reward_EMA = cov_reward_curr_next
                        else:
                            self.cov_reward_EMA = self.config.evaluator_EMA_decay * cov_reward_curr_next + (1 - self.config.evaluator_EMA_decay) * self.cov_reward_EMA
                        mask_diagonal_cov_reward = torch.eye(self.cov_reward_EMA.shape[0], dtype=torch.bool, device=self.cov_reward_EMA.device)
                        self.cov_reward_EMA[mask_diagonal_cov_reward] = self.cov_reward_EMA[mask_diagonal_cov_reward].clamp_min(1e-8)

                    rand_mask = torch.rand(size_batch, len_seq - 1, device=state_rep_next.device).reshape(-1)

                    if "generate" in self.config.evaluator_relabeling_strategy:
                        succ_imagined = self.wm.rssm.img_step(outputs['post'], data["action"], prob_rand=0.0) # NOTE: deter is the same for post and prior if observing x, since only the posterior is rolled forward
                        stoch_next_imagined = succ_imagined["stoch"][:, :-1].reshape(size_batch, len_seq - 1, -1)
                        deter_next_imagined = succ_imagined["deter"][:, :-1].to(self.evaluator.device)

                        stoch_next_imagined = stoch_next_imagined.reshape(-1, stoch_next_imagined.shape[-1])
                        deter_next_imagined = deter_next_imagined.reshape(-1, deter_next_imagined.shape[-1])
                        feat_next_imagined = torch.cat([stoch_next_imagined, deter_next_imagined], -1)
                        feat_next_imagined = feat_next_imagined.reshape(-1, feat_next_imagined.shape[-1])

                        if self.config.evaluator_h_deter_realism: # these are only one step realisms
                            deter_curr_next_imagined = torch.cat([deter_curr, deter_next_imagined], -1)
                            diff_deter_next_targ_T = (deter_curr_next_imagined - deter_curr_next).T
                            Lmah_deter_imagined_persample = (diff_deter_next_targ_T * torch.linalg.solve(self.cov_deter_EMA, diff_deter_next_targ_T)).sum(0).sqrt_()
                            assert not torch.isnan(Lmah_deter_imagined_persample).any()
                        if self.config.evaluator_h_value_realism:
                            value_next_imagined = self._task_behavior._target_critic(feat_next_imagined).mode.reshape(-1, 1)
                            value_curr_next_imagined = torch.cat([value_curr, value_next_imagined], -1)
                            diff_value_next_targ_T = (value_curr_next_imagined - value_curr_next).T
                            Lmah_value_imagined_persample = (diff_value_next_targ_T * torch.linalg.solve(self.cov_value_EMA, diff_value_next_targ_T)).sum(0).sqrt_()
                            assert not torch.isnan(Lmah_value_imagined_persample).any()
                        if self.config.evaluator_h_pcont_realism:
                            pcont_next_imagined = self.wm.heads['discount'](feat_next_imagined).mean.reshape(-1, 1)
                            pcont_curr_next_imagined = torch.cat([pcont_curr, pcont_next_imagined], -1)
                            diff_pcont_next_targ_T = (pcont_curr_next_imagined - pcont_curr_next).T
                            Lmah_pcont_imagined_persample = (diff_pcont_next_targ_T * torch.linalg.solve(self.cov_pcont_EMA, diff_pcont_next_targ_T)).sum(0).sqrt_()
                            assert not torch.isnan(Lmah_pcont_imagined_persample).any()
                        if self.config.evaluator_h_reward_realism:
                            reward_next_imagined = self.wm.heads['reward'](feat_next_imagined).mode.reshape(-1, 1)
                            reward_curr_next_imagined = torch.cat([reward_curr, reward_next_imagined], -1)
                            diff_reward_next_targ_T = (reward_curr_next_imagined - reward_curr_next).T
                            Lmah_reward_imagined_persample = (diff_reward_next_targ_T * torch.linalg.solve(self.cov_reward_EMA, diff_reward_next_targ_T)).sum(0).sqrt_()
                            assert not torch.isnan(Lmah_reward_imagined_persample).any()
                        if self.config.evaluator_rep == "stoch":
                            state_rep_imagined = stoch_next_imagined
                        elif self.config.evaluator_rep == "deter":
                            state_rep_imagined = deter_next_imagined
                        elif self.config.evaluator_rep == "stoch+deter":
                            state_rep_imagined = torch.cat([stoch_next_imagined, deter_next_imagined], -1)
                        else:
                            raise NotImplementedError("")
                    if "episode" in self.config.evaluator_relabeling_strategy:
                        indices0 = torch.repeat_interleave(torch.arange(size_batch, device=state["deter"].device), len_seq - 1)
                        indices1 = torch.rand(size_batch, len_seq - 1, device=state["deter"].device).argsort(-1).reshape(-1)
                        indices2 = torch.repeat_interleave(torch.arange(len_seq - 1, device=state["deter"].device).reshape(1, -1), size_batch, dim=0).reshape(-1)
                        mask_episode_coincident = (indices1 == indices2)
                        state_rep_next_shuffled = state_rep_next.reshape(size_batch, len_seq - 1, -1)[indices0, indices1].reshape(state_rep_next.shape[0], -1)
                        if self.config.evaluator_h_deter_realism:
                            deter_next_shuffled = deter_next.reshape(size_batch, len_seq - 1, -1)[indices0, indices1].reshape(deter_next.shape[0], -1)
                            deter_curr_next_shuffled = torch.cat([deter_curr, deter_next_shuffled], -1)
                            diff_deter_next_shuffled_T = (deter_curr_next_shuffled - deter_curr_next).T
                            Lmah_deter_episode_persample = (diff_deter_next_shuffled_T * torch.linalg.solve(self.cov_deter_EMA, diff_deter_next_shuffled_T)).sum(0).sqrt_()
                            Lmah_deter_episode_persample[mask_episode_coincident] = torch.nan
                        if self.config.evaluator_h_value_realism:
                            value_next_shuffled = value_next.reshape(size_batch, len_seq - 1, -1)[indices0, indices1].reshape(value_next.shape[0], -1)
                            value_curr_next_shuffled = torch.cat([value_curr, value_next_shuffled], -1)
                            diff_value_next_shuffled_T = (value_curr_next_shuffled - value_curr_next).T
                            Lmah_value_episode_persample = (diff_value_next_shuffled_T * torch.linalg.solve(self.cov_value_EMA, diff_value_next_shuffled_T)).sum(0).sqrt_()
                            Lmah_value_episode_persample[mask_episode_coincident] = torch.nan
                        if self.config.evaluator_h_pcont_realism:
                            pcont_next_shuffled = pcont_next.reshape(size_batch, len_seq - 1, -1)[indices0, indices1].reshape(pcont_next.shape[0], -1)
                            pcont_curr_next_shuffled = torch.cat([pcont_curr, pcont_next_shuffled], -1)
                            diff_pcont_next_shuffled_T = (pcont_curr_next_shuffled - pcont_curr_next).T
                            Lmah_pcont_episode_persample = (diff_pcont_next_shuffled_T * torch.linalg.solve(self.cov_pcont_EMA, diff_pcont_next_shuffled_T)).sum(0).sqrt_()
                            Lmah_pcont_episode_persample[mask_episode_coincident] = torch.nan
                        if self.config.evaluator_h_reward_realism:
                            reward_next_shuffled = reward_next.reshape(size_batch, len_seq - 1, -1)[indices0, indices1].reshape(reward_next.shape[0], -1)
                            reward_curr_next_shuffled = torch.cat([reward_curr, reward_next_shuffled], -1)
                            diff_reward_next_shuffled_T = (reward_curr_next_shuffled - reward_curr_next).T
                            Lmah_reward_episode_persample = (diff_reward_next_shuffled_T * torch.linalg.solve(self.cov_reward_EMA, diff_reward_next_shuffled_T)).sum(0).sqrt_()
                            Lmah_reward_episode_persample[mask_episode_coincident] = torch.nan
                    if self.config.evaluator_relabeling_strategy == "generate":
                        state_rep_targ = state_rep_imagined.clone()
                        mask_nextastarg = torch.zeros(size_batch, len_seq - 1, dtype=torch.bool, device=state_rep_next.device).reshape(-1)
                        mask_imagined = torch.ones(size_batch, len_seq - 1, dtype=torch.bool, device=state_rep_next.device).reshape(-1)
                        mask_episode = torch.zeros(size_batch, len_seq - 1, dtype=torch.bool, device=state_rep_next.device).reshape(-1)
                        if self.config.evaluator_h_deter_realism:
                            Lmah_deter_targ_persample = Lmah_deter_imagined_persample.clone()
                        if self.config.evaluator_h_value_realism:
                            Lmah_value_targ_persample = Lmah_value_imagined_persample.clone()
                        if self.config.evaluator_h_pcont_realism:
                            Lmah_pcont_targ_persample = Lmah_pcont_imagined_persample.clone()
                        if self.config.evaluator_h_reward_realism:
                            Lmah_reward_targ_persample = Lmah_reward_imagined_persample.clone()
                    elif self.config.evaluator_relabeling_strategy == "next+generate":
                        state_rep_targ = state_rep_imagined.clone()
                        mask_nextastarg = rand_mask < 0.5
                        state_rep_targ[mask_nextastarg] = state_rep_next[mask_nextastarg]
                        if self.config.evaluator_h_deter_realism:
                            Lmah_deter_targ_persample = Lmah_deter_imagined_persample.clone()
                            Lmah_deter_targ_persample[mask_nextastarg] = 0.0
                        if self.config.evaluator_h_value_realism:
                            Lmah_value_targ_persample = Lmah_value_imagined_persample.clone()
                            Lmah_value_targ_persample[mask_nextastarg] = 0.0
                        if self.config.evaluator_h_pcont_realism:
                            Lmah_pcont_targ_persample = Lmah_pcont_imagined_persample.clone()
                            Lmah_pcont_targ_persample[mask_nextastarg] = 0.0
                        if self.config.evaluator_h_reward_realism:
                            Lmah_reward_targ_persample = Lmah_reward_imagined_persample.clone()
                            Lmah_reward_targ_persample[mask_nextastarg] = 0.0
                        mask_episode = torch.zeros(size_batch, len_seq - 1, dtype=torch.bool, device=state_rep_next.device).reshape(-1)
                        mask_imagined = ~mask_nextastarg
                    elif self.config.evaluator_relabeling_strategy == "episode+generate":
                        state_rep_targ = state_rep_imagined.clone()
                        mask_imagined = rand_mask >= 0.5
                        mask_nextastarg = torch.zeros_like(rand_mask, dtype=torch.bool)
                        
                        mask_episode = rand_mask < 0.5
                        state_rep_targ[mask_episode] = state_rep_next_shuffled[mask_episode]

                        if self.config.evaluator_h_deter_realism:
                            Lmah_deter_targ_persample = Lmah_deter_imagined_persample.clone()
                            Lmah_deter_targ_persample[mask_episode] = Lmah_deter_episode_persample[mask_episode] 
                            Lmah_deter_targ_persample[mask_episode_coincident] = 0
                        if self.config.evaluator_h_value_realism:
                            Lmah_value_targ_persample = Lmah_value_imagined_persample.clone()
                            Lmah_value_targ_persample[mask_episode] = Lmah_value_episode_persample[mask_episode]
                            Lmah_value_targ_persample[mask_episode_coincident] = 0
                        if self.config.evaluator_h_pcont_realism:
                            Lmah_pcont_targ_persample = Lmah_pcont_imagined_persample.clone()
                            Lmah_pcont_targ_persample[mask_episode] = Lmah_pcont_episode_persample[mask_episode]
                            Lmah_pcont_targ_persample[mask_episode_coincident] = 0
                        if self.config.evaluator_h_reward_realism:
                            Lmah_reward_targ_persample = Lmah_reward_imagined_persample.clone()
                            Lmah_reward_targ_persample[mask_episode] = Lmah_reward_episode_persample[mask_episode]
                            Lmah_reward_targ_persample[mask_episode_coincident] = 0
                    elif self.config.evaluator_relabeling_strategy == "next+episode+generate":
                        state_rep_targ = state_rep_imagined.clone()
                        mask_imagined = rand_mask >= 0.6
                        mask_nextastarg = rand_mask < 0.20
                        state_rep_targ[mask_nextastarg] = state_rep_next[mask_nextastarg]
                        if self.config.evaluator_h_deter_realism:
                            Lmah_deter_targ_persample = Lmah_deter_imagined_persample.clone()
                            Lmah_deter_targ_persample[mask_nextastarg] = 0.0
                        if self.config.evaluator_h_value_realism:
                            Lmah_value_targ_persample = Lmah_value_imagined_persample.clone()
                            Lmah_value_targ_persample[mask_nextastarg] = 0.0
                        if self.config.evaluator_h_pcont_realism:
                            Lmah_pcont_targ_persample = Lmah_pcont_imagined_persample.clone()
                            Lmah_pcont_targ_persample[mask_nextastarg] = 0.0
                        if self.config.evaluator_h_reward_realism:
                            Lmah_reward_targ_persample = Lmah_reward_imagined_persample.clone()
                            Lmah_reward_targ_persample[mask_nextastarg] = 0.0

                        mask_episode = (rand_mask >= 0.2) & (rand_mask < 0.6)
                        state_rep_targ[mask_episode] = state_rep_next_shuffled[mask_episode]

                        if self.config.evaluator_h_deter_realism:
                            Lmah_deter_targ_persample[mask_episode] = Lmah_deter_episode_persample[mask_episode] # torch.inf # 
                            Lmah_deter_targ_persample[mask_episode_coincident] = 0
                        if self.config.evaluator_h_value_realism:
                            Lmah_value_targ_persample[mask_episode] = Lmah_value_episode_persample[mask_episode]
                            Lmah_value_targ_persample[mask_episode_coincident] = 0
                        if self.config.evaluator_h_pcont_realism:
                            Lmah_pcont_targ_persample[mask_episode] = Lmah_pcont_episode_persample[mask_episode]
                            Lmah_pcont_targ_persample[mask_episode_coincident] = 0
                        if self.config.evaluator_h_reward_realism:
                            Lmah_reward_targ_persample[mask_episode] = Lmah_reward_episode_persample[mask_episode]
                            Lmah_reward_targ_persample[mask_episode_coincident] = 0
                    else:
                        raise NotImplementedError("")
                    
                    mask_deter_close, mask_value_close, mask_pcont_close, mask_reward_close = None, None, None, None
                    mask_reached = torch.ones(size_batch, len_seq - 1, dtype=torch.bool, device=state_rep_next.device).reshape(-1)
                    # assert "generate" in self.config.evaluator_relabeling_strategy and "episode" in self.config.evaluator_relabeling_strategy
                    if self.config.evaluator_h_deter_realism:
                        threshold_deter = self.config.evaluator_h_deter_threshold # torch.quantile(Lmah_deter_episode_persample[Lmah_deter_episode_persample > 0], 0.05).item()
                        mask_deter_close = (Lmah_deter_targ_persample < threshold_deter).reshape(-1)
                        mask_reached &= mask_deter_close
                    if self.config.evaluator_h_value_realism:
                        threshold_value = self.config.evaluator_h_value_threshold
                        mask_value_close = (Lmah_value_targ_persample < threshold_value).reshape(-1)
                        mask_reached &= mask_value_close
                    if self.config.evaluator_h_pcont_realism:
                        threshold_pcont = self.config.evaluator_h_pcont_threshold
                        mask_pcont_close = (Lmah_pcont_targ_persample < threshold_pcont).reshape(-1)
                        mask_reached &= mask_pcont_close
                    if self.config.evaluator_h_reward_realism:
                        threshold_reward = self.config.evaluator_h_reward_threshold
                        mask_reward_close = (Lmah_reward_targ_persample < threshold_reward).reshape(-1)
                        mask_reached &= mask_reward_close

                # if mask_nextastarg.any():
                #     assert mask_reached[mask_nextastarg].all()
                # if "episode" in self.config.evaluator_relabeling_strategy:
                #     flag_train_evaluator = mask_reached[~mask_nextastarg & ~mask_episode_coincident].any() # if all chaotic, don't waste time training
                # else:
                #     flag_train_evaluator = mask_reached[~mask_nextastarg].any()
                flag_train_evaluator = True
                if flag_train_evaluator:
                    loss, loss_avg, distance_curr, target_discount_distance, norm_grad = self.evaluator.train(source_curr=state_rep_curr.detach(), action_curr=action_curr.detach(), source_next=state_rep_next.detach(), action_next=action_next.detach(), target=state_rep_targ.detach(), mask_reached=mask_reached, mask_done=mask_done, increment_counter=True, update_params=update_params, flag_debug=flag_debug, weights=None)
 
                if update_params and flag_debug and np.random.rand() < 0.05:
                    with torch.no_grad():
                        metrics['evaluator_train/ratio_reached'] = mask_reached.float().mean().item()
                        if mask_imagined.any():
                            metrics['evaluator_train/ratio_imagined_reached'] = mask_reached[mask_imagined].float().mean().item()
                        if mask_episode.any():
                            metrics['evaluator_train/ratio_episode_reached'] = mask_reached[mask_episode].float().mean().item()
                        if "generate" in self.config.evaluator_relabeling_strategy:
                            if self.config.evaluator_h_deter_realism:
                                metrics['evaluator_train/ratio_imagined_deter_close'] = (Lmah_deter_imagined_persample < threshold_deter).float().mean().item()
                                histograms_record['evaluator_train/Lmah_deter_imagined'] = Lmah_deter_imagined_persample.clamp_max(2 * threshold_deter)
                            if self.config.evaluator_h_value_realism:
                                metrics['evaluator_train/ratio_imagined_value_close'] = (Lmah_value_imagined_persample < threshold_value).float().mean().item()
                                histograms_record['evaluator_train/Lmah_value_imagined'] = Lmah_value_imagined_persample.clamp_max(2 * threshold_value)
                            if self.config.evaluator_h_pcont_realism:
                                metrics['evaluator_train/ratio_imagined_pcont_close'] = (Lmah_pcont_imagined_persample < threshold_pcont).float().mean().item()
                                histograms_record['evaluator_train/Lmah_pcont_imagined'] = Lmah_pcont_imagined_persample.clamp_max(2 * threshold_pcont)
                            if self.config.evaluator_h_reward_realism:
                                metrics['evaluator_train/ratio_imagined_reward_close'] = (Lmah_reward_imagined_persample < threshold_reward).float().mean().item()
                                histograms_record['evaluator_train/Lmah_reward_imagined'] = Lmah_reward_imagined_persample.clamp_max(2 * threshold_reward)
                        if "episode" in self.config.evaluator_relabeling_strategy:
                            if self.config.evaluator_h_deter_realism:
                                Lmah_deter_episode_persample_noncoincident = Lmah_deter_episode_persample[~mask_episode_coincident]
                                metrics['evaluator_train/ratio_episode_deter_close'] = (Lmah_deter_episode_persample_noncoincident < threshold_deter).float().mean().item()
                                histograms_record['evaluator_train/Lmah_deter_episode'] = Lmah_deter_episode_persample_noncoincident.clamp_max(4 * threshold_deter)
                            if self.config.evaluator_h_value_realism:
                                Lmah_value_episode_persample_noncoincident = Lmah_value_episode_persample[~mask_episode_coincident]
                                metrics['evaluator_train/ratio_episode_value_close'] = (Lmah_value_episode_persample_noncoincident < threshold_value).float().mean().item()
                                histograms_record['evaluator_train/Lmah_value_episode'] = Lmah_value_episode_persample_noncoincident.clamp_max(4 * threshold_value)
                            if self.config.evaluator_h_pcont_realism:
                                Lmah_pcont_episode_persample_noncoincident = Lmah_pcont_episode_persample[~mask_episode_coincident]
                                metrics['evaluator_train/ratio_episode_pcont_close'] = (Lmah_pcont_episode_persample_noncoincident < threshold_pcont).float().mean().item()
                                histograms_record['evaluator_train/Lmah_pcont_episode'] = Lmah_pcont_episode_persample_noncoincident.clamp_max(4 * threshold_pcont)
                            if self.config.evaluator_h_reward_realism:
                                Lmah_reward_episode_persample_noncoincident = Lmah_reward_episode_persample[~mask_episode_coincident]
                                metrics['evaluator_train/ratio_episode_reward_close'] = (Lmah_reward_episode_persample_noncoincident < threshold_reward).float().mean().item()
                                histograms_record['evaluator_train/Lmah_reward_episode'] = Lmah_reward_episode_persample_noncoincident.clamp_max(4 * threshold_reward)
                        # if "generate" in self.config.evaluator_relabeling_strategy and "episode" in self.config.evaluator_relabeling_strategy:
                            # if self.config.evaluator_h_deter_realism:
                            #     metrics['evaluator_train/threshold_deter'] = threshold_deter
                            # if self.config.evaluator_h_value_realism:
                            #     metrics['evaluator_train/threshold_value'] = threshold_value
                            # if self.config.evaluator_h_pcont_realism:
                            #     metrics['evaluator_train/threshold_pcont'] = threshold_pcont
                            # if self.config.evaluator_h_reward_realism:
                            #     metrics['evaluator_train/threshold_reward'] = threshold_reward

                        if flag_train_evaluator:
                            metrics['evaluator_train/loss_avg'] = loss_avg.item()
                            metrics['evaluator_train/res_dist2updatetarget'] = torch.abs(distance_curr[mask_imagined].clamp(1, self.evaluator.atoms) - target_discount_distance[mask_imagined].clamp(1, self.evaluator.atoms)).mean().item()
                            metrics['evaluator_train/norm_grad'] = norm_grad.item()
                            histogram_imagined_d1 = self.evaluator(source=state_rep_curr, target=state_rep_imagined, action=action_curr, type_output="logits", use_targetnet=False).softmax(dim=-1)
                            distances_imagined_d1 = histogram_imagined_d1 @ (1.0 + torch.arange(histogram_imagined_d1.shape[-1], dtype=torch.float32, device=histogram_imagined_d1.device))
                            metrics['evaluator_validate/feasibility_imagined_d1'] = histogram_imagined_d1[:, 0].mean().item()
                            metrics['evaluator_validate/distances_imagined_d1'] = distances_imagined_d1.mean().item()

                            # NOTE: multistep metrics
                            sources, targets, actions, dists_target = [state_rep_curr], [state_rep_next], [action_curr], [torch.full([size_batch * (len_seq - 1), 1], 1.0, dtype=torch.float32, device=state_rep_curr.device)]
                            state_rep_curr_reshaped = state_rep_curr.reshape(size_batch, len_seq - 1, -1)
                            state_rep_next_reshaped = state_rep_next.reshape(size_batch, len_seq - 1, -1)
                            action_curr_reshaped = action_curr.reshape(size_batch, len_seq - 1, -1)
                            for i in range(1, min(len_seq - 1, self.evaluator.atoms)):
                                sources.append(state_rep_curr_reshaped[:, :-i].reshape(-1, state_rep_curr_reshaped.shape[-1]))
                                targets.append(state_rep_next_reshaped[:, i:].reshape(-1, state_rep_next_reshaped.shape[-1]))
                                actions.append(action_curr_reshaped[:, :-i].reshape(-1, action_curr_reshaped.shape[-1]))
                                dists_target.append(torch.full([size_batch * (len_seq - 1 - i), 1], i + 1, dtype=torch.float32, device=state_rep_curr.device))
                            sources = torch.cat(sources, 0)
                            targets = torch.cat(targets, 0)
                            actions = torch.cat(actions, 0)
                            dists_target = torch.cat(dists_target, 0).reshape(-1)
                            histogram_multistep = self.evaluator(source=sources, target=targets, action=actions, type_output="logits", use_targetnet=False).softmax(dim=-1)
                            distances_multistep = histogram_multistep @ (1.0 + torch.arange(histogram_multistep.shape[-1], dtype=torch.float32, device=histogram_multistep.device))
                            res_distances_multistep = torch.abs(distances_multistep - dists_target)
                            assert (res_distances_multistep <= self.evaluator.atoms - 1).all()
                            res_distances_avg = []
                            for i in range(min(len_seq - 1, self.evaluator.atoms)):
                                mask_multistep = (dists_target == i + 1)
                                res_distances_di = res_distances_multistep[mask_multistep].mean().item()
                                metrics[f'evaluator_validate/res_distance_d{i + 1:d}'] = res_distances_di
                                metrics[f'evaluator_validate/dist_hat_d{i + 1:d}'] = distances_multistep[mask_multistep].mean().item()
                                res_distances_avg.append(res_distances_di)
                            metrics['evaluator_validate/res_distances_multistep_avg'] = np.mean(res_distances_avg)
        random.setstate(rd_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)

        metrics.update(mets)
        start = outputs['post']
        if self.config.pred_discount:  # Last step could be terminal.
            # start = tf.nest.map_structure(lambda x: x[:, :-1], start)
            start = {k: v[:, :-1].detach() for k, v in start.items()}
        else:
            start = common.dict_detach(start)  # detach post
        reward = lambda f, s, a: self.wm.heads['reward'](f).mode
        metrics_AC, histograms_AC = self._task_behavior.train(self.wm, start, reward, evaluator=self.evaluator, update_params=update_params)
        metrics.update(metrics_AC)
        # if self.config.expl_behavior != 'greedy':
        #     if self.config.pred_discount:
        #         # data = tf.nest.map_structure(lambda x: x[:, :-1], data)
        #         # outputs = tf.nest.map_structure(lambda x: x[:, :-1], outputs)
        #         data = {k: v[:, :-1] for k, v in data.items()}  # FIXME check this
        #         outputs = {k: v[:, :-1] for k, v in outputs.items()}
        #     mets = self._expl_behavior.train(start, outputs, data)[-1] #FIXME outputs have previous graph from wm
        #     metrics.update({'expl_' + key: value for key, value in mets.items()})
        histograms_record.update(histograms_AC)
        return common.dict_detach(state), metrics, histograms_record

    def report(self, data, state=None):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=common.ENABLE_FP16):
                return {'openl': self.wm.video_pred(data, state).cpu().permute(0, 2, 3, 1).numpy()}  # T,H,W,C needs to be numpy

    def init_optimizers(self):
        wm_modules = [self.wm.encoder.parameters(), self.wm.rssm.parameters(),
                      *[head.parameters() for head in self.wm.heads.values()]]
        self.wm.model_opt = common.Optimizer('model', wm_modules, **self.config.model_opt)

        self._task_behavior.actor_opt = common.Optimizer('actor', self._task_behavior.actor.parameters(),
                                                         **self.config.actor_opt)
        self._task_behavior.critic_opt = common.Optimizer('critic', self._task_behavior.critic.parameters(),
                                                          **self.config.critic_opt)





class WorldModel(common.Module):
    def __init__(self, step, config):
        super(WorldModel, self).__init__()
        self.step = step
        self.config = config
        self.rssm = common.RSSM(**config.rssm)
        self.heads = {}
        self.shape = (1 if config.grayscale else 3,) + config.image_size

        out_shape=self.shape
        if tuple(config.encoder['keys']) == ('flatten',):
            out_shape = [np.prod(self.shape)]

        self.encoder = common.ConvEncoder(**config.encoder)

        self.heads = nn.ModuleDict({
            'image': common.ConvDecoder(out_shape, **config.decoder),
            'reward': common.MLP([], **config.reward_head),
        })
        if config.pred_discount:
            self.heads.update({
                'discount': common.MLP([], **config.discount_head)
            })
        for name in config.grad_heads:
            assert name in self.heads, name
        self.model_opt = common.EmptyOptimizer()

    def train(self, data, state=None):
        with torch.cuda.amp.autocast(enabled=common.ENABLE_FP16):
            self.zero_grad(set_to_none=True)  # delete grads

            model_loss, state, outputs, metrics = self.loss(data, state)

        # Backward passes under autocast are not recommended.
        self.model_opt.backward(model_loss)
        metrics.update(self.model_opt.step(model_loss))
        metrics['model_loss'] = model_loss.item()
        return state, outputs, metrics

    def loss(self, data, state=None):
        data = self.preprocess(data)
        embed = self.encoder(data)
        post, prior = self.rssm.observe(embed, data['action'], state) # NOTE: using posterior to roll forward ...
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl) # prior is Z_hat, post is Z
        likes = {}
        losses = {'kl': kl_loss}
        feat = self.rssm.get_feat(post)
        for name, head in self.heads.items():
            grad_head = (name in self.config.grad_heads)
            inp = feat if grad_head else feat.detach()

            like = head(inp).log_prob(data[name])

            likes[name] = like
            losses[name] = -like.mean()
        model_loss = sum(
            self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items())
        outs = dict(
            embed=embed, feat=feat, post=post,
            prior=prior, likes=likes, kl=kl_value)  # stop propagating gradients? for now is okay, disabled in agent
        with torch.no_grad():
            metrics = {f'{name}_loss': value.item() for name, value in losses.items()}
            metrics['model_kl'] = kl_value.mean().item()
            metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean().item()
            metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean().item()
        return model_loss, post, outs, metrics

    def imagine(self, policy, start, horizon):
        flatten = lambda x: x.reshape(-1, *x.shape[2:])
        start = {k: flatten(v) for k, v in start.items()}

        def step(state, _):
            # state, _, _ = prev
            feat = self.rssm.get_feat(state)

            action = policy(feat.detach()).sample()
            # action = policy(feat.detach()).mean  # for testing DEBUG
            with torch.no_grad():
                succ = self.rssm.img_step(state, action, prob_rand=self.config.img_step_prob_rand)

            return succ, feat, action

        # feat = 0 * self.rssm.get_feat(start)
        # action = policy(feat).mode  # NOt used?
        # succs, feats, actions = common.static_scan(
        #     step, tf.range(horizon), (start, feat, action))

        succs, feats, actions = common.sequence_scan(step, start, np.arange(horizon))
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succs.items()}

        with torch.no_grad():  # FIXME
            if 'discount' in self.heads:
                discount = self.heads['discount'](feats).mean
            else:
                discount = self.config.discount * torch.ones_like(feats[..., 0])

        return feats, states, actions, discount

    def preprocess(self, obs):
        # dtype = prec.global_policy().compute_dtype #everything is casted in data_loader next_batch
        # obs = obs.copy()  # doesnt clone, but forces to clone on equals
        obs['image'] = obs['image'] / 255.0 - 0.5

        clip_function = lambda x: x if self.config.clip_rewards == 'identity' else getattr(torch, self.config.clip_rewards)(x)
        obs['reward'] = clip_function(obs['reward'])
        if 'discount' in obs:
            obs['discount'] = obs['discount'] * self.config.discount
        return obs

    def video_pred(self, data, state=None):
        '''
        FIXME do transforms on cpu
        Log images reconstructions come from this function

        '''

        data = self.preprocess(data)
        embed = self.encoder(data)
        states, _ = self.rssm.observe(embed[:6, :5], data['action'][:6, :5], state)
        recon = self.heads['image'](self.rssm.get_feat(states)).mode[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.rssm.imagine(data['action'][:6, 5:], init)
        openl = self.heads['image'](self.rssm.get_feat(prior)).mode

        # select 6 envs, do 5 frames from data, rest reconstruct from dataset
        # so if dataset has 50 frames, 5 initial are real, 50-5 are imagined

        # should do on cpu?
        recon = recon.cpu()
        openl = openl.cpu()
        truth = data['image'][:6].cpu() + 0.5

        if len(recon.shape)==3: #flat
            recon = recon.reshape(*recon.shape[:-1],*self.shape)
            openl = openl.reshape(*openl.shape[:-1],*self.shape)
            truth = truth.reshape(*truth.shape[:-1],*self.shape)


        model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)  # time
        error = (model - truth + 1) / 2
        video = torch.cat([truth, model, error], 3)  # on H
        B, T, C, H, W = video.shape  # batch, time, height,width, channels
        return video.permute(1, 2, 3, 0, 4).reshape(T, C, H, B * W)


#
class ActorCritic(common.Module):
    def __init__(self, config, step, num_actions):
        super(ActorCritic, self).__init__()
        self.config = config
        self.step = step
        self.num_actions = num_actions
        self.actor = common.MLP(num_actions, **config.actor)
        self.critic = common.MLP([], **config.critic)
        if config.slow_target:
            self._target_critic = common.MLP([], **config.critic)
            self._updates = 0
        else:
            self._target_critic = self.critic
        self.actor_opt = common.EmptyOptimizer()
        self.critic_opt = common.EmptyOptimizer()

        self.once = 0

    def train(self, world_model, start, reward_fn, evaluator=None, update_params=True):
        metrics = {}
        hor = self.config.imag_horizon
        with torch.cuda.amp.autocast(enabled=common.ENABLE_FP16):
            # delete grads
            world_model.zero_grad(set_to_none=True)
            self.actor.zero_grad(set_to_none=True)
            self.critic.zero_grad(set_to_none=True)

            feat, state, action, disc = world_model.imagine(self.actor, start, hor)
            # NOTE: here the things are only imagined for training the ActorCritic
            histograms_AC = {}

            rd_state = random.getstate()
            np_state = np.random.get_state()
            torch_state = torch.random.get_rng_state()
            mask_reject, mask_reject_autoregr, mask_reject_frominit = None, None, None
            with torch.random.fork_rng():
                if evaluator is None or not update_params:
                    flag_record = False
                else:
                    flag_record = np.random.rand() < 0.02
                    if (self.config.evaluator_reject and self.step.value >= self.config.evaluator_reject_start) or flag_record:
                        with torch.no_grad():
                            action_reshaped = action.reshape(hor, self.config.dataset["batch"], self.config.dataset["length"] - 1, *action.shape[2:])
                            feat_reshaped = feat.reshape(hor, self.config.dataset["batch"], self.config.dataset["length"] - 1, *feat.shape[2:])
                            # NOTE: the last one in the observed horizon is pre-removed, losses only established on s_currs
                            action_curr = action_reshaped[:-1, :, :]
                            action_init = action_reshaped[[0], :, :]
                            if self.config.evaluator_rep == "stoch":
                                state_rep_curr = feat_reshaped[:-1, :, :, :self.config.rssm["stoch"] * self.config.rssm["discrete"]]
                                state_rep_targ = feat_reshaped[1:, :, :, :self.config.rssm["stoch"] * self.config.rssm["discrete"]]
                                state_rep_init = feat_reshaped[[0], :, :, :self.config.rssm["stoch"] * self.config.rssm["discrete"]]
                            elif self.config.evaluator_rep == "deter":
                                state_rep_curr = feat_reshaped[:-1, :, :, self.config.rssm["stoch"] * self.config.rssm["discrete"]:]
                                state_rep_targ = feat_reshaped[1:, :, :, self.config.rssm["stoch"] * self.config.rssm["discrete"]:]
                                state_rep_init = feat_reshaped[[0], :, :, self.config.rssm["stoch"] * self.config.rssm["discrete"]:]
                            elif self.config.evaluator_rep == "stoch+deter":
                                state_rep_curr = feat_reshaped[:-1, :, :]
                                state_rep_targ = feat_reshaped[1:, :, :] # NOTE: one step rollout, targ is next
                                state_rep_init = feat_reshaped[[0], :, :]
                            else:
                                raise NotImplementedError()
                            
                            state_rep_curr = state_rep_curr.reshape(-1, *state_rep_curr.shape[-1:]).to(evaluator.device)
                            action_curr = action_curr.reshape(-1, *action_curr.shape[-1:]).to(evaluator.device)
                            state_rep_targ = state_rep_targ.reshape(-1, *state_rep_targ.shape[-1:]).to(evaluator.device)

                            logits_autoregr = evaluator(source=state_rep_curr, target=state_rep_targ, action=action_curr, type_output="logits", use_targetnet=False)
                            histograms_autoregr = logits_autoregr.reshape(hor - 1, self.config.dataset["batch"], self.config.dataset["length"] - 1, -1).softmax(dim=-1)
                            distances_autoregr = histograms_autoregr @ (1.0 + torch.arange(histograms_autoregr.shape[-1], dtype=torch.float32, device=histograms_autoregr.device))
                            feasibility_autoregr = histograms_autoregr[:, :, :, 0]
                            mask_reject_autoregr = (feasibility_autoregr < self.config.evaluator_threshold_reject_autoregr)

                            if flag_record:
                                metrics['evaluator_infer/ratio_reject_autoregr'] = mask_reject_autoregr.float().mean().item()
                                metrics['evaluator_infer/ratio_reject_autoregr_5'] = (feasibility_autoregr < 0.05).float().mean().item()
                                metrics['evaluator_infer/ratio_reject_autoregr_10'] = (feasibility_autoregr < 0.1).float().mean().item()
                                metrics['evaluator_infer/ratio_reject_autoregr_20'] = (feasibility_autoregr < 0.2).float().mean().item()
                                metrics['evaluator_infer/ratio_reject_autoregr_25'] = (feasibility_autoregr < 0.25).float().mean().item()
                                metrics['evaluator_infer/ratio_reject_autoregr_33'] = (feasibility_autoregr < 0.33).float().mean().item()
                                metrics['evaluator_infer/ratio_reject_autoregr_50'] = (feasibility_autoregr < 0.5).float().mean().item()

                                histograms_AC[f'evaluator_infer/feasibility_autoregr'] = feasibility_autoregr.reshape(-1)
                                histograms_AC[f'evaluator_infer/distances_autoregr'] = distances_autoregr.reshape(-1)

                            logits_frominit = evaluator(source=torch.repeat_interleave(state_rep_init, hor - 1, dim=0).reshape(-1, state_rep_init.shape[-1]), target=state_rep_targ, action=torch.repeat_interleave(action_init, hor - 1, dim=0).reshape(-1, action_init.shape[-1]), type_output="logits", use_targetnet=False)
                            histograms_frominit = logits_frominit.reshape(hor - 1, self.config.dataset["batch"], self.config.dataset["length"] - 1, -1).softmax(dim=-1)
                            # assert (histograms_frominit[0] == histograms_autoregr[0]).all()
                            distances_frominit = histograms_frominit @ (1.0 + torch.arange(histograms_frominit.shape[-1], dtype=torch.float32, device=histograms_frominit.device))
                            # assert (distances_autoregr[0] == distances_frominit[0]).all()
                            if flag_record:
                                metrics[f'evaluator_infer/res_distance_frominit_avg'] = torch.abs(distances_frominit - 1.0 - torch.arange(hor - 1, dtype=torch.float32, device=distances_frominit.device).reshape(-1, 1, 1)).mean().item() # direct P2P distances

                                histograms_AC[f'evaluator_infer/res_distance_frominit'] = torch.abs(distances_frominit - 1.0 - torch.arange(distances_frominit.shape[0], dtype=torch.float32, device=distances_frominit.device).reshape(-1, 1, 1)).reshape(-1)
                            
                            feasibility_cum_frominit = histograms_frominit[:, :, :, :-1].sum(-1)
                            if flag_record:
                                histograms_AC[f'evaluator_infer/feasibility_cum_frominit'] = feasibility_cum_frominit.reshape(-1)

                            mask_reject_frominit = (feasibility_cum_frominit < self.config.evaluator_threshold_reject_frominit)
                            if flag_record:
                                metrics['evaluator_infer/ratio_reject_frominit'] = mask_reject_frominit.float().mean().item()
                                metrics['evaluator_infer/ratio_reject_frominit_50'] = (feasibility_cum_frominit < 0.5).float().mean().item()
                                metrics['evaluator_infer/ratio_reject_frominit_75'] = (feasibility_cum_frominit < 0.75).float().mean().item()
                                metrics['evaluator_infer/ratio_reject_frominit_80'] = (feasibility_cum_frominit < 0.8).float().mean().item()
                                metrics['evaluator_infer/ratio_reject_frominit_90'] = (feasibility_cum_frominit < 0.9).float().mean().item()
                                metrics['evaluator_infer/ratio_reject_frominit_95'] = (feasibility_cum_frominit < 0.95).float().mean().item()
                                metrics['evaluator_infer/ratio_reject_frominit_99'] = (feasibility_cum_frominit < 0.99).float().mean().item()

                                histograms_frominit_arranged = histograms_frominit.reshape(hor - 1, -1, evaluator.atoms).permute(1, 0 ,2)
                                feasibility_frominit_exact = histograms_frominit_arranged[torch.repeat_interleave(torch.arange(histograms_frominit_arranged.shape[0], dtype=torch.long), hor - 1), torch.arange(hor - 1).repeat(histograms_frominit_arranged.shape[0]), torch.arange(hor - 1).repeat(histograms_frominit_arranged.shape[0])].reshape(histograms_frominit_arranged.shape[0], hor - 1).T.reshape(feasibility_autoregr.shape)
                                # assert (feasibility_autoregr[0] == feasibility_frominit_exact[0]).all()
                                res_distance_autoregr_avg = []
                                for idx_hor in range(hor - 1):
                                    feasibility_autoregr_step = feasibility_autoregr[idx_hor, :]
                                    distances_autoregr_step = distances_autoregr[:idx_hor + 1, :].sum(0)
                                    histograms_AC[f'evaluator_infer/feasibility_autoregr_d{idx_hor + 1:d}'] = feasibility_autoregr_step.reshape(-1)
                                    res_distance_autoregr_this_d = torch.abs(distances_autoregr_step.reshape(-1) - 1.0 - float(idx_hor)).mean().item()
                                    metrics[f'evaluator_infer/res_distance_autoregr_d{idx_hor + 1:d}'] = res_distance_autoregr_this_d # cumulative distances with the one-step autoregressive rollouts
                                    res_distance_autoregr_avg.append(res_distance_autoregr_this_d)
                                    feasibility_frominit_step = feasibility_frominit_exact[idx_hor, :]
                                    feasibility_cum_frominit_step = feasibility_cum_frominit[idx_hor, :]
                                    distances_frominit_step = distances_frominit[idx_hor, :]
                                    mask_reject_frominit_step = mask_reject_frominit[idx_hor, :]
                                    metrics[f'evaluator_infer/ratio_reject_frominit_d{idx_hor + 1:d}'] = mask_reject_frominit_step.float().mean().item()
                                    histograms_AC[f'evaluator_infer/feasibility_frominit_exactly_d{idx_hor + 1:d}'] = feasibility_frominit_step.reshape(-1)
                                    metrics[f'evaluator_infer/feasibility_frominit_exactly_d{idx_hor + 1:d}_avg'] = histograms_AC[f'evaluator_infer/feasibility_frominit_exactly_d{idx_hor + 1:d}'].mean().item()
                                    histograms_AC[f'evaluator_infer/feasibility_frominit_cum_d{idx_hor + 1:d}'] = feasibility_cum_frominit_step.reshape(-1)
                                    metrics[f'evaluator_infer/feasibility_frominit_cum_d{idx_hor + 1:d}_avg'] = histograms_AC[f'evaluator_infer/feasibility_frominit_cum_d{idx_hor + 1:d}'].mean().item()
                                    metrics[f'evaluator_infer/res_distance_frominit_d{idx_hor + 1:d}'] = torch.abs(distances_frominit_step.reshape(-1) - 1.0 - float(idx_hor)).mean().item() # direct P2P distances
                                metrics['evaluator_infer/res_distance_autoregr_avg'] = np.mean(res_distance_autoregr_avg)
                            mask_reject = mask_reject_autoregr | mask_reject_frominit
                            if flag_record:
                                metrics['evaluator_infer/ratio_reject'] = mask_reject.float().mean().item()
                            if not self.config.evaluator_reject or self.step.value < self.config.evaluator_reject_start:
                                mask_reject, mask_reject_autoregr, mask_reject_frominit = None, None, None
            random.setstate(rd_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)

            with torch.no_grad():  # FIXME only for mode reinforce is enabled
                reward = reward_fn(feat, state, action)
                target, weight, mets1 = self.target(feat, action, reward, disc, mask_reject=mask_reject, flag_debug=flag_record)  # weight doesnt prop
                rd_state = random.getstate()
                np_state = np.random.get_state()
                torch_state = torch.random.get_rng_state()
                if mask_reject_frominit is not None: # NOTE: this is very harsh
                    weight[:-1] *= (~mask_reject_frominit).flatten(1).float() # don't update states that are not reachable at all!
                random.setstate(rd_state)
                np.random.set_state(np_state)
                torch.set_rng_state(torch_state)
                target = target.detach()
                weight = weight.detach()
            actor_loss, mets2 = self.actor_loss(feat, action, target, weight)
            critic_loss, mets3 = self.critic_loss(feat, action, target, weight)

        # Backward passes under autocast are not recommended.
        if mask_reject is None or not torch.isnan(actor_loss):
            self.actor_opt.backward(actor_loss, retain_graph=True)
            metrics.update(self.actor_opt.step(actor_loss))
        if mask_reject is None or not torch.isnan(critic_loss):
            self.critic_opt.backward(critic_loss)
            metrics.update(self.critic_opt.step(critic_loss))
            
        metrics.update(**mets1, **mets2, **mets3)
        self.update_slow_target()
        return metrics, histograms_AC

    def actor_loss(self, feat, action, target, weight):
        metrics = {}
        policy = self.actor(feat.detach())  # FIXME why would we do this again? use previous from imagine? deactivate grads?
        mask_nan_target = torch.isnan(target)
        if self.config.actor_grad == 'dynamics':
            objective = target
            objective[mask_nan_target] = 0.0
            objective = objective.detach()
        elif self.config.actor_grad == 'reinforce':
            baseline = self.critic(feat[:-1]).mode
            advantage = (target - baseline).detach()  # note here nothing props through critic
            advantage[mask_nan_target] = 0.0
            assert not torch.isnan(advantage).any()
            objective = policy.log_prob(action)[:-1] * advantage.detach()  # note, no grads to action, only to policy
            assert not torch.isnan(objective).any()
        elif self.config.actor_grad == 'both':
            baseline = self.critic(feat[:-1]).mode
            advantage = (target - baseline).detach()
            advantage[mask_nan_target] = 0.0
            objective = policy.log_prob(action)[:-1] * advantage.detach()
            mix = common.schedule(self.config.actor_grad_mix, self.step)
            objective = mix * target + (1 - mix) * objective
            metrics['actor_grad_mix'] = mix
        else:
            raise NotImplementedError(self.config.actor_grad)
        ent = policy.entropy()
        ent_scale = common.schedule(self.config.actor_ent, self.step)
        if mask_nan_target.any():
            actor_loss = -1.0 * ((weight[:-1] * objective)[~mask_nan_target]).mean() - ent_scale * (weight[:-1] * ent[:-1]).mean() # let entropy loss be the same
        else:
            objective += ent_scale * ent[:-1]
            actor_loss = -1.0 * (weight[:-1] * objective).mean()

        if not mask_nan_target.all():
            # metrics['actor_ent'] = ent[:-1][~mask_nan_target].mean().item()
            metrics['actor_ent'] = ent[:-1].mean().item()
        metrics['actor_ent_scale'] = ent_scale

        # debug
        mse_logits = (policy.orig_logits ** 2).mean()
        metrics['actor_logits_mse'] = mse_logits.item()

        metrics['z_actor_logits_policy_max'] = policy.orig_logits.max().item()  # unnormalized logits
        metrics['z_actor_logits_policy_min'] = policy.orig_logits.min().item()

        if torch.any(policy.logits.isnan()):
            print("actor logits nan")
        if torch.any(policy.logits.isinf()):
            print("actor logits inf")

        return actor_loss, metrics

    def critic_loss(self, feat, action, target, weight):
        # print("critic loss, ",feat.shape,target.shape,weight.shape)
        dist = self.critic(feat[:-1])
        mask_nan = torch.isnan(target)
        target_clone = target.clone()
        target_clone[torch.isnan(target)] = 0.0
        log_probs = dist.log_prob(target_clone.detach())
        log_probs[mask_nan] = torch.nan
        neg_critic_loss_elements = log_probs * weight[:-1]
        # assert (mask_nan == torch.isnan(neg_critic_loss_elements)).all()
        critic_loss = -1.0 * neg_critic_loss_elements[~mask_nan].mean() # log_prob of normal(out,1) is - mse
        metrics = {'critic': dist.mode.mean().item()}
        return critic_loss, metrics

    def target(self, feat, action, reward, disc, mask_reject=None, flag_debug=False):
        # print("target feat type",type(feat),"args:",feat.shape,reward.shape,disc.shape)
        # reward = tf.cast(reward, tf.float32) #FIXME verify casts
        # disc = tf.cast(disc, tf.float32)
        metrics = {}
        value = self._target_critic(feat).mode
        metrics['critic_slow'] = value.mean().item()
        target = common.lambda_return(
            reward[:-1], value[:-1], disc[:-1],
            bootstrap=value[-1], lambda_=self.config.discount_lambda, axis=0, mask_reject=mask_reject)
        
        # weight -> discount**i, where weight[0] = discount**1
        weight = torch.cumprod(torch.cat([torch.ones_like(disc[:1]), disc[:-1]], 0).detach(), 0)

        if mask_reject is not None and flag_debug:
            if mask_reject.all():
                metrics['evaluator_infer/ratio_target_changed'] = 1.0
                metrics['evaluator_infer/diff_target'] = torch.nan
                metrics['evaluator_infer/diff_target_min'] = torch.nan
                metrics['evaluator_infer/diff_target_max'] = torch.nan
                metrics['evaluator_infer/diff_target_L1'] = torch.nan
                metrics['evaluator_infer/diff_target_L1_max'] = torch.nan
            elif mask_reject.any():
                mask_relevant = mask_reject.any(0).reshape(-1)
                mask_reject_relevant = mask_reject.flatten(1).T[mask_relevant]
                if not mask_reject_relevant.all():
                    target_original = common.lambda_return(reward[:-1], value[:-1], disc[:-1], bootstrap=value[-1], lambda_=self.config.discount_lambda, axis=0, mask_reject=None)
                    diff_target = target - target_original
                    diff_target_L1 = torch.abs(diff_target)
                    hor = diff_target.shape[0]
                    for idx_hor in range(hor):
                        metrics[f'evaluator_infer/diff_target_L1_d{idx_hor + 1:d}'] = diff_target_L1[idx_hor, :].nanmean().item()
                    diff_target_relevant = diff_target.permute(1, 0)[mask_relevant]
                    diff_target_L1_relevant = diff_target_L1.permute(1, 0)[mask_relevant]
                    metrics['evaluator_infer/diff_target'] = diff_target_relevant[~mask_reject_relevant].mean().item()
                    metrics['evaluator_infer/diff_target_min'] = diff_target_relevant[~mask_reject_relevant].min().item()
                    metrics['evaluator_infer/diff_target_max'] = diff_target_relevant[~mask_reject_relevant].max().item()
                    metrics['evaluator_infer/diff_target_L1'] = diff_target_L1_relevant[~mask_reject_relevant].mean().item()
                    metrics['evaluator_infer/diff_target_L1_max'] = diff_target_L1_relevant[~mask_reject_relevant].max().item()
                    metrics['evaluator_infer/ratio_target_changed'] = (target != target_original).float().mean().item()
            else:
                metrics['evaluator_infer/ratio_target_changed'] = 0.0
                
        metrics['reward_mean'] = reward.mean().item()
        metrics['reward_std'] = reward.std().item()
        if mask_reject is None:
            metrics['critic_target'] = target.mean().item()
        elif not mask_reject.all():
            metrics['critic_target'] = target[~torch.isnan(target)].mean().item()
        else:
            metrics['critic_target'] = torch.nan
        metrics['discount'] = disc.mean().item()
        return target, weight, metrics

    def update_slow_target(self):  # polyak update
        if self.config.slow_target:
            if self._updates % self.config.slow_target_update == 0:
                mix = 1.0 if self._updates == 0 else float(
                    self.config.slow_target_fraction)
                for s, d in zip(self.critic.parameters(), self._target_critic.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
