from typing import Dict, Callable, Optional, Iterable

import numpy as np

from tqdm import tqdm

from cpprb import ReplayBuffer, PrioritizedReplayBuffer

class HindsightReplayBuffer:
    def __init__(
        self,
        size: int,
        env_dict: Dict,
        max_episode_len: int,
        *,
        goal_func: Optional[Callable] = None,
        goal_shape: Optional[Iterable[int]] = None,
        state: str = "obs",
        action: str = "act",
        next_state: str = "next_obs",
        strategy_primary: str = "future",
        strategy_secondary: str = "episode",
        pertask_mixrate: Optional[Iterable[float]] = [0.0, 0.5],
        additional_goals: int = 4,
        prioritized=True,
        gamma=1.0,
        no_goal=False, # NOTE(H): fallback to normal ER
        num_envs=1,
        rb_pertask_sample=None,
        ctx=None,
        **kwargs,
    ):
        self.max_episode_len = max_episode_len
        self.goal_func = goal_func
        self.gamma = gamma # to calculate the return_before_discounted and return_entire_discounted
        self.no_goal = bool(no_goal)
        self.num_envs = num_envs

        self.state = state
        self.action = action
        self.next_state = next_state

        self.additional_goals = additional_goals

        assert strategy_primary in ["episode", "future", "pertask"]
        assert strategy_secondary in ["episode", "future", "pertask", None]

        self.strategy_primary = strategy_primary
        self.strategy_secondary = strategy_secondary
        self.pertask_mixrate = pertask_mixrate

        if self.strategy_primary == "pertask":
            self.pertask_mixrate[0] = 1.0
        if self.pertask_mixrate[0] == 1.0:
            self.strategy_primary = "pertask"
        if self.pertask_mixrate[0] == 0.0:
            assert self.strategy_primary != "pertask"
        if self.pertask_mixrate[0] > 0:
            assert self.strategy_primary != "future" # will be a total mess
        
        if self.strategy_secondary is not None:
            if self.strategy_secondary == "pertask":
                self.pertask_mixrate[1] = 1.0
            if self.pertask_mixrate[1] == 1.0:
                self.strategy_secondary = "pertask"
            if self.pertask_mixrate[1] == 0.0:
                assert self.strategy_secondary != "pertask"
            if self.pertask_mixrate[1] > 0:
                assert self.strategy_secondary != "future" # will be a total mess
            
        self.prioritized = prioritized

        if goal_shape:
            goal_dict = {**env_dict[state], "shape": goal_shape}
            self.goal_shape = np.array(goal_shape, ndmin=1)
        else:
            goal_dict = env_dict[state]
            self.goal_shape = np.array(env_dict[state].get("shape", 1), ndmin=1)

        if self.no_goal:
            dict_init = {**env_dict}
        else:
            dict_init = {**env_dict, "goal": goal_dict}
        
        if self.pertask_mixrate[0] > 0 or (self.strategy_secondary is not None and self.pertask_mixrate[1] > 0):
            dict_init[f"idx_env"] = {"shape": 1, "dtype": int} # NOTE(H): if we use separate buffers for each env, we won't have the global priorities
            env_dict["idx_env"] = {"shape": 1, "dtype": int}
        
        if self.strategy_secondary is not None:
            dict_init["goal_secondary"] = goal_dict
        self.dict_rb_init = dict_init

        if ctx is not None:
            RB = ctx.PrioritizedReplayBuffer if self.prioritized else ctx.ReplayBuffer
        else:
            RB = PrioritizedReplayBuffer if self.prioritized else ReplayBuffer
        if self.prioritized and ctx is not None:
            self.rb = RB(size, self.dict_rb_init, check_for_update=False, **kwargs)
        else:
            self.rb = RB(size, self.dict_rb_init, **kwargs)
        if ctx is not None:
            self.episode_rb = ctx.ReplayBuffer(self.max_episode_len, env_dict)
        else:
            self.episode_rb = ReplayBuffer(self.max_episode_len, env_dict)

        self.rng = np.random.default_rng()

        if self.pertask_mixrate[0] > 0 or (self.strategy_secondary is not None and self.pertask_mixrate[1] > 0):
            if rb_pertask_sample is None:
                self.rb_pertask_sample = self
                self.rbs_pertask = []
                for idx_env in tqdm(range(self.num_envs), desc="create individual rbs for each env"):
                    dict_init_pertask = {"obs": self.dict_rb_init["obs"]}
                    size_rb_individual = size // (2 * self.additional_goals)
                    if ctx is not None:
                        self.rbs_pertask.append(ctx.ReplayBuffer(size_rb_individual, dict_init_pertask, **kwargs))
                    else:
                        self.rbs_pertask.append(ReplayBuffer(size_rb_individual, dict_init_pertask, **kwargs))
            else:
                self.rb_pertask_sample = rb_pertask_sample

    def add(self, **kwargs):
        r"""Add transition(s) into replay buffer.

        Multple sets of transitions can be added simultaneously.

        Parameters
        ----------
        **kwargs : array like or float or int
            Transitions to be stored.
        """
        if self.episode_rb.get_stored_size() >= self.max_episode_len:
            raise ValueError("Exceed Max Episode Length")
        self.episode_rb.add(**kwargs)

    def sample(self, batch_size: int, **kwargs):
        r"""Sample the stored transitions randomly with specified size

        Parameters
        ----------
        batch_size : int
            sampled batch size

        Returns
        -------
        dict of ndarray
            Sampled batch transitions, which might contains
            the same transition multiple times.
        """
        return self.rb.sample(batch_size, **kwargs)

    def on_episode_end(self):
        r"""
        Terminate the current episode and set hindsight goals
        """
        episode_len = self.episode_rb.get_stored_size()
        if episode_len == 0:
            return None

        trajectory = self.episode_rb.get_all_transitions()

        if self.no_goal:
            self.rb.add(**trajectory)
            self.episode_rb.clear()
            self.rb.on_episode_end()
            return None

        num_samples_needed = self.additional_goals * episode_len

        if self.pertask_mixrate[0] > 0 or (self.strategy_secondary is not None and self.pertask_mixrate[1] > 0):
            idx_env = int(trajectory["idx_env"][0])
            assert (trajectory["idx_env"] == idx_env).all()
            self.rb_pertask_sample.rbs_pertask[idx_env].add(obs=trajectory[self.state][[0]])
            self.rb_pertask_sample.rbs_pertask[idx_env].add(obs=trajectory[self.next_state])
        
        if self.pertask_mixrate[0] > 0:
            if np.random.rand() < self.pertask_mixrate[0] and self.rb_pertask_sample.rbs_pertask[idx_env].get_stored_size() > num_samples_needed:
                traj_sampled = self.rb_pertask_sample.rbs_pertask[idx_env].sample(num_samples_needed)
                possible_goals_primary = traj_sampled["obs"]
            else:
                possible_goals_primary = trajectory[self.next_state] # fallback
        else:
            possible_goals_primary = trajectory[self.next_state]
        
        if self.strategy_secondary is not None:
            if self.pertask_mixrate[1] > 0:
                if np.random.rand() < self.pertask_mixrate[1] and self.rb_pertask_sample.rbs_pertask[idx_env].get_stored_size() > num_samples_needed:
                    traj_sampled = self.rb_pertask_sample.rbs_pertask[idx_env].sample(num_samples_needed)
                    possible_goals_secondary = traj_sampled["obs"]
                else:
                    possible_goals_secondary = trajectory[self.next_state] # fallback
            else:
                possible_goals_secondary = trajectory[self.next_state]

        data2submit, data2submit_secondary = [], []
        idx = np.full((self.additional_goals, episode_len), -1, dtype=np.int64)
        for i in range(episode_len):
            low = i if "future" in self.strategy_primary else 0
            transition = {}
            for key in list(trajectory.keys()):
                transition[key] = trajectory[key][i]
            idx[:, i] = np.sort(self.rng.integers(low=low, high=possible_goals_primary.shape[0], size=self.additional_goals))
            for j in range(self.additional_goals):
                idx_targ = idx[j, i]
                if self.goal_func is None:
                    goal = possible_goals_primary[idx_targ]
                else:
                    goal = self.goal_func(possible_goals_primary[idx_targ])
                data2submit.append(transition | {"goal": goal})

        if self.strategy_secondary is not None:
            idx = np.full((self.additional_goals, episode_len), -1, dtype=np.int64)
            for i in range(episode_len):
                low = i if "future" in self.strategy_secondary else 0
                transition = {}
                for key in list(trajectory.keys()):
                    transition[key] = trajectory[key][i]
                idx[:, i] = np.sort(self.rng.integers(low=low, high=possible_goals_secondary.shape[0], size=self.additional_goals))
                for j in range(self.additional_goals):
                    idx_targ = idx[j, i]
                    if self.goal_func is None:
                        goal_secondary = possible_goals_secondary[idx_targ]
                    else:
                        goal_secondary = self.goal_func(possible_goals_secondary[idx_targ])
                    data2submit_secondary.append({"goal_secondary": goal_secondary})

        trajectory2submit = {}
        for key in self.dict_rb_init.keys():
            to_concat = []
            for entry in data2submit:
                if key in entry.keys():
                    to_concat.append(entry[key].reshape(self.dict_rb_init[key]["add_shape"]))
            if data2submit_secondary is not None:
                for entry in data2submit_secondary:
                    if key in entry.keys():
                        to_concat.append(entry[key].reshape(self.dict_rb_init[key]["add_shape"]))
            trajectory2submit[key] = np.concatenate(to_concat, 0)
        self.rb.add(**trajectory2submit)
        self.episode_rb.clear()
        self.rb.on_episode_end()

    def clear(self):
        """
        Clear replay buffer
        """
        self.rb.clear()
        self.episode_rb.clear()

    def get_stored_size(self):
        """
        Get stored size

        Returns
        -------
        int
            stored size
        """
        return self.rb.get_stored_size()

    def get_buffer_size(self):
        """
        Get buffer size

        Returns
        -------
        int
            buffer size
        """
        return self.rb.get_buffer_size()

    def get_all_transitions(self, shuffle: bool = False):
        r"""
        Get all transitions stored in replay buffer.

        Parameters
        ----------
        shuffle : bool, optional
            When ``True``, transitions are shuffled. The default value is ``False``.

        Returns
        -------
        transitions : dict of numpy.ndarray
            All transitions stored in this replay buffer.
        """
        return self.rb.get_all_transitions(shuffle)

    def update_priorities(self, indexes, priorities):
        """
        Update priorities

        Parameters
        ----------
        indexes : array_like
            indexes to update priorities
        priorities : array_like
            priorities to update

        Raises
        ------
        TypeError: When ``indexes`` or ``priorities`` are ``None``
        ValueError: When this buffer is constructed with ``prioritized=False``
        """
        if not self.prioritized:
            raise ValueError("Buffer is constructed without PER")

        self.rb.update_priorities(indexes, priorities)

    def get_max_priority(self):
        """
        Get max priority

        Returns
        -------
        float
            Max priority of stored priorities

        Raises
        ------
        ValueError: When this buffer is constructed with ``prioritized=False``
        """
        if not self.prioritized:
            raise ValueError("Buffer is constructed without PER")

        return self.rb.get_max_priority()
