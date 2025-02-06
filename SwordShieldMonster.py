"""
                        +x (dim 0)
                    0--------------→
                    |       3
                    |       ↑
                    |       |
    +y (dim 1)      |  2 ←--+--→ 0
                    |       |
                    |       ↓
                    ↓       1

width * height
"""

import numpy as np, torch
import minigrid
from minigrid import *
from utils import dijkstra, floyd_warshall
import copy

    
def obs2context(obs, type="compact"):
    # SwordShieldMonster.check_obs_validity(obs)
    if len(obs.shape) == 3:
        obs = obs[None, :, :, :]
    _, width, height, _ = obs.shape
    slice_color = obs[:, :, :, 1]
    _, x_monster, y_monster = torch.where(slice_color == COLOR_TO_IDX["green"])
    _, x_sword, y_sword = torch.where(slice_color == COLOR_TO_IDX["purple"])
    _, x_shield, y_shield = torch.where(slice_color == COLOR_TO_IDX["blue"])
    if type == "compact":
        return torch.stack([x_monster, y_monster, x_sword, y_sword, x_shield, y_shield], -1).float()
    elif type == "onehot":
        x_monster = torch.nn.functional.one_hot(x_monster, num_classes=width)
        y_monster = torch.nn.functional.one_hot(y_monster, num_classes=height)
        x_sword = torch.nn.functional.one_hot(x_sword, num_classes=width)
        y_sword = torch.nn.functional.one_hot(y_sword, num_classes=height)
        x_shield = torch.nn.functional.one_hot(x_shield, num_classes=width)
        y_shield = torch.nn.functional.one_hot(y_shield, num_classes=height)
        return torch.cat([x_monster, y_monster, x_sword, y_sword, x_shield, y_shield], -1)
    else:
        raise NotImplementedError("")

def obs2agentmap(obs, ignore_dir=False):
    slice = obs[:, :, 0]
    if ignore_dir:
        return slice == OBJECT_TO_IDX["agent"]
    else:
        return slice == OBJECT_TO_IDX["agent"], obs[:, :, -1]
    
def obs2swordmap(obs):
    slice = obs[:, :, 0]
    return slice == OBJECT_TO_IDX["sword"]

def obs2shieldmap(obs):
    slice = obs[:, :, 0]
    return slice == OBJECT_TO_IDX["shield"]

def obs2monstermap(obs):
    slice = obs[:, :, 0]
    return slice == OBJECT_TO_IDX["monster"]

STR_MISSION = "get the sword and shield and kill the monster"

class Grid(minigrid.Grid):
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache = {}

    def encode(self, vis_mask=None, ignore_color=False, ignore_dir=False):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros(
            (self.width, self.height, 3 - int(ignore_color) - int(ignore_dir)),
            dtype="uint8",
        )

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = OBJECT_TO_IDX["empty"]
                        if not ignore_color:
                            array[i, j, 1] = 0
                        if not ignore_dir:
                            array[i, j, -1] = 0

                    else:
                        v_encoded = v.encode()
                        array[i, j, 0] = v_encoded[0]
                        if not ignore_color:
                            array[i, j, 1] = v_encoded[1]
                        if not ignore_dir:
                            array[i, j, -1] = v_encoded[-1]
        return array

    def render(self, tile_size, agent_pos, agent_dir=None, highlight_mask=None, obs=None):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if obs is None:
            width, height = self.width, self.height
        else:
            width, height = obs.shape[0], obs.shape[1]

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(width, height), dtype=bool)

        # Compute the total grid size
        width_px = width * tile_size
        height_px = height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        if obs is not None:
            lava_map = (obs[:, :, 0] == OBJECT_TO_IDX["lava"]).squeeze()
            map_agent, agent_dir = obs2agentmap(obs)  # NOTE(H): lots of agents potentially, lol
            map_sword = obs2swordmap(obs)
            map_shield = obs2shieldmap(obs)
            map_monster = obs2monstermap(obs)


        # Render the grid
        for j in range(0, height):
            for i in range(0, width):
                if obs is None:
                    cell = self.get(i, j)
                else:
                    if lava_map[i, j]:
                        cell = Lava()
                    elif map_sword[i, j]:
                        cell = Sword()
                    elif map_shield[i, j]:
                        cell = Shield()
                    elif map_monster[i, j]:
                        cell = Monster()
                    else:
                        cell = None

                if agent_pos is None:
                    agent_here = False
                    agent_dir = None
                else:
                    agent_here = map_agent[i, j]
                tile_img = self.render_tile(
                    cell,
                    agent_dir=agent_dir[i, j] if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def render_tile(cls, obj, agent_dir=None, highlight=False, tile_size=TILE_PIXELS, subdivs=3):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key = (agent_dir, highlight, tile_size)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj is not None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:
            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
            fill_coords(img, tri_fn, (255, 0, 0))

        # Highlight the cell if needed
        if highlight:
            highlight_img2(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

def swordshield2x(sword_acquired=False, shield_acquired=False):
    """use an integer to denote of the sword and the shield are acquired, taking values from 0 to 3"""
    if isinstance(sword_acquired, np.ndarray):
        assert isinstance(shield_acquired, np.ndarray)
        assert sword_acquired.shape == shield_acquired.shape
        return sword_acquired.astype(np.int64) * 2 + shield_acquired.astype(np.int64)
    else:
        return int(sword_acquired) * 2 + int(shield_acquired)
    
def x2swordshield(x): # validated
    if isinstance(x, np.ndarray):
        assert (x >= 0).all() and (x <= 3).all()
        return (x // 2).astype(bool), (x % 2).astype(bool)
    else:
        assert x >= 0 and x <= 3
        return bool(x // 2), bool(x % 2)

class SwordShieldMonster(MiniGridEnv_Custom):
    def __init__(
        self,
        width=8,
        height=8,
        lava_density_range=[0.3, 0.4],
        gamma=0.99,
        uniform_init=False,
        stochasticity=0.0,
        singleton=True,
    ):
        self.name_game = "SwordShieldMonster"
        lava_density = np.random.uniform(lava_density_range[0], lava_density_range[1])
        self.transposed = False
        self.obs_goal = None
        self.singleton = singleton # NOTE(H): if false, use a cross as a target, 5 states per target (max)
        self.total_possible_lava = width * height - 2 * width
        self.max_lava_blocks = int(self.total_possible_lava * lava_density)
        self.agent_start_dir = np.random.randint(0, 4)
        self.pos_agent_init = (np.random.randint(0, width), height - 1)
        
        if np.random.rand() <= 0.5:
            self.pos_agent_init = (0, np.random.randint(0, height))
            self.pos_monster = (width - 1, np.random.randint(0, height))
        else:
            self.pos_agent_init = (width - 1, np.random.randint(0, height))
            self.pos_monster = (0, np.random.randint(0, height))
        self.agent_dir, self.dir_agent = 0, 0
        self.sword_acquired, self.shield_acquired = False, False
        
        self.pos_sword = (np.random.randint(1, width - 1), np.random.randint(0, height))
        self.pos_shield = (np.random.randint(1, width - 1), np.random.randint(0, height))
        while self.pos_sword[0] == self.pos_shield[0] and self.pos_sword[1] == self.pos_shield[1]: # make sure not coincident
            self.pos_sword = (np.random.randint(1, width - 1), np.random.randint(0, height))
            self.pos_shield = (np.random.randint(1, width - 1), np.random.randint(0, height))
        self.width, self.height = width, height
        self.ignore_dir = False  # only v2 could change this for now
        self.generate_map()
        assert not self.lava_map[self.pos_sword[0], self.pos_sword[1]] and not self.lava_map[self.pos_shield[0], self.pos_shield[1]]
        mission_space = MissionSpace(mission_func=lambda: STR_MISSION)
        MiniGridEnv.__init__(
            self,
            width=width,
            height=height,
            max_steps=128,
            see_through_walls=True,
            agent_view_size=int(2 * max(width, height) - 1),
            mission_space=mission_space,
        )
        self.gamma = gamma
        self.render_mode = "rgb_array"
        self.init_DP_info()
        self.uniform_init = uniform_init
        assert stochasticity >= 0.0 and stochasticity <= 1.0
        self.stochasticity = stochasticity

    def collect_states_reachable(self):
        if self.DP_info["lava_map"] is None:
            self.init_DP_assets()
        if self.DP_info["P"] is None:
            self.collect_transition_probs()
        if self.DP_info["A"] is None:
            self.collect_state_adjacency()

        state_start = self.ijxd2state(self.pos_agent_init[0], self.pos_agent_init[1], 0)
        ret = dijkstra(self.DP_info["A"], state_start)
        states_reachable_from_start = [state_start]
        for target_state in range(len(ret)):
            distance = ret[target_state]
            if distance != np.inf and state_start != target_state:
                states_reachable_from_start.append(target_state)
        
        states_reachable_from_start = sorted(states_reachable_from_start)
        self.DP_info["states_reachable"] = states_reachable_from_start

        state_goal = self.obs2state(self.obs_goal)
        if state_goal not in states_reachable_from_start:
            print(f"pos_agent_init: {self.pos_agent_init}, pos_monster: {self.pos_monster}")
            print(f"pos_sword: {self.pos_sword}, pos_shield: {self.pos_shield}")
            print(self.DP_info["lava_map"].astype(int).transpose())
            raise RuntimeError("goal state not reachable from start state")
        assert self.ijxd2state(self.pos_sword[0], self.pos_sword[1], swordshield2x(False, False)) not in self.DP_info["states_reachable"]
        assert self.ijxd2state(self.pos_sword[0], self.pos_sword[1], swordshield2x(False, True)) not in self.DP_info["states_reachable"]
        assert self.ijxd2state(self.pos_shield[0], self.pos_shield[1], swordshield2x(False, False)) not in self.DP_info["states_reachable"]
        assert self.ijxd2state(self.pos_shield[0], self.pos_shield[1], swordshield2x(True, False)) not in self.DP_info["states_reachable"]
        
        omega_states = np.zeros(self.DP_info["P"].shape[-1], dtype=bool)
        for idx_state in range(self.DP_info["P"].shape[-1]):
            if (self.DP_info["P"][:, idx_state, idx_state] == 1.0).all():
                omega_states[idx_state] = True
        omega_states_existent = omega_states[self.DP_info["states_reachable"]]
        self.DP_info["omega_states"] = omega_states
        self.DP_info["omega_states_existent"] = omega_states_existent

    @property
    def x_curr(self):
        return swordshield2x(self.sword_acquired, self.shield_acquired)

    def load_layout_from_obs(self, obs, pos_sword=None, pos_shield=None):
        SwordShieldMonster.check_obs_validity(obs)
        assert len(obs.shape) == 3
        width, height, _ = obs.shape
        assert self.width == width and self.height == height
        slice = obs[:, :, 0]
        self.lava_map = np.zeros_like(slice, dtype=bool)
        # agent_pos, agent_dir = None, None
        self.pos_sword, self.pos_shield, self.pos_monster = pos_sword, pos_shield, None
        self.sword_acquired, self.shield_acquired = True, True
        self.agent_start_dir = 0
        for i in range(width):
            for j in range(height):
                if slice[i, j] == OBJECT_TO_IDX["agent"]:
                    if self.ignore_dir:
                        self.agent_pos, self.agent_dir = (i, j), 0
                    else:
                        self.agent_pos, self.agent_dir = (i, j), int(obs[i, j, -1])
                    if obs[i, j, 1] == COLOR_TO_IDX["yellow"]:
                        self.lava_map[i, j] = True
                    elif obs[i, j, 1] == COLOR_TO_IDX["green"]:
                        self.pos_monster = (i, j)
                elif slice[i, j] == OBJECT_TO_IDX["monster"]:
                    self.pos_monster = (i, j)
                elif slice[i, j] == OBJECT_TO_IDX["lava"]:
                    self.lava_map[i, j] = True
                elif slice[i, j] == OBJECT_TO_IDX["sword"]:
                    if pos_sword is not None:
                        assert pos_sword[0] == i and pos_sword[0] == j
                    self.pos_sword = (i, j)
                    self.sword_acquired = False
                elif slice[i, j] == OBJECT_TO_IDX["shield"]:
                    if pos_shield is not None:
                        assert pos_shield[0] == i and pos_shield[0] == j
                    self.pos_shield = (i, j)
                    self.shield_acquired = False
        assert self.agent_pos is not None and self.pos_monster is not None
        assert self.pos_sword is not None and self.pos_shield is not None
        self.pos_agent_init = (width - 1 - self.pos_monster[0], np.random.randint(0, height))
        self._gen_grid(width, height)
        self.init_DP_info()
        self.collect_states_reachable()
        self.obs_curr = self.gen_fullyobservable_obs()

    def init_DP_info(self):
        self.DP_info = {
            "ijxd_targ": np.array([*self.pos_monster, 3]),
            "pos_monster": self.pos_monster,
            "pos_sword": self.pos_sword,
            "pos_shield": self.pos_shield,
            "num_states": None,
            "lava_map": None,
            "Q_optimal": None,
            "Q_random": None,
            "r": None,
            "P": None,
            "A": None,
            "state_target_tuples": None,
            "obses_all": None,
            "obses_all_processed": None,
            "states_reachable": None,
        }

    def gen_fullyobservable_obs(self):
        return self.draw_obs_with_agent(self.agent_pos[0], self.agent_pos[1], swordshield2x(self.sword_acquired, self.shield_acquired), self.agent_dir)

    def generate_random_path(self, epsilon=0.35, start=None, end=None):
        if start is None:
            start = self.pos_agent_init
        if end is None:
            end = self.pos_monster
        pos_curr = np.copy(np.array(start))
        while pos_curr[0] != end[0] or pos_curr[1] != end[1]:
            move_x = np.random.rand() < 0.5
            move_random = np.random.rand() < epsilon
            move_curr = np.array([0, 0])
            if move_x:
                if move_random:
                    move_curr[0] = np.random.randint(low=-1, high=2)
                else:
                    diff_x = end[0] - pos_curr[0]
                    if diff_x != 0:
                        move_curr[0] = np.sign(diff_x)
                    else:
                        diff_x = 0
            else:
                if move_random:
                    move_curr[1] = np.random.randint(low=-1, high=2)
                else:
                    diff_y = end[1] - pos_curr[1]
                    if diff_y != 0:
                        move_curr[1] = np.sign(diff_y)
                    else:
                        diff_y = 0
            pos_next = pos_curr + move_curr
            if pos_next[0] == end[0] and pos_next[1] == end[1]:
                break
            else:
                pos_next[0] = np.clip(pos_next[0], 0, self.width - 1)
                pos_next[1] = np.clip(pos_next[1], 0, self.height - 1)
                if pos_next[0] != self.pos_monster[0] or pos_next[1] != self.pos_monster[1]: # NOTE(H): make sure the path to sword or shield is not blocked by monster
                    pos_curr = pos_next
                    self.lava_map[pos_curr[0], pos_curr[1]] = False
                # else:
                #     print("path intercepted by monster, redirecting path")
                # # print(f"start: {start}, pos_curr: {pos_curr}, end: {end}")

    def reset_gen_map(self):
        self.lava_map = np.zeros((self.width, self.height), dtype=bool)
        self.lava_map[1 : self.width - 1, 0 : self.height] = True
        self.lava_map[self.pos_agent_init[0], self.pos_agent_init[1]] = False
        self.lava_map[self.pos_sword[0], self.pos_sword[1]] = False
        self.lava_map[self.pos_shield[0], self.pos_shield[1]] = False
        self.lava_map[self.pos_monster[0], self.pos_monster[1]] = False

    def generate_map(self):
        while True:
            self.reset_gen_map()
            self.generate_random_path(start=self.pos_shield, end=self.pos_sword)
            self.generate_random_path(start=self.pos_agent_init, end=self.pos_sword)
            self.generate_random_path(start=self.pos_agent_init, end=self.pos_shield)
            self.generate_random_path(start=self.pos_sword, end=self.pos_monster)
            self.generate_random_path(start=self.pos_shield, end=self.pos_monster)
            # if np.random.rand() > 0.5:
            #     self.generate_random_path(start=self.pos_agent_init, end=self.pos_sword)
            # else:
            #     self.generate_random_path(start=self.pos_agent_init, end=self.pos_shield)
            # if np.random.rand() > 0.5:
            #     self.generate_random_path(start=self.pos_sword, end=self.pos_monster)
            # else:
            #     self.generate_random_path(start=self.pos_shield, end=self.pos_monster)
            remaining_lava_blocks = int(np.sum(self.lava_map))
            if remaining_lava_blocks > self.max_lava_blocks:
                break

        if remaining_lava_blocks > self.max_lava_blocks:
            lava_indices = np.nonzero(self.lava_map)
            lava_indices_x = lava_indices[0]
            lava_indices_y = lava_indices[1]
            perm = np.random.permutation(lava_indices_x.shape[0])
            lava_indices_x = lava_indices_x[perm]
            lava_indices_y = lava_indices_y[perm]
            for i in range(int(remaining_lava_blocks - self.max_lava_blocks)):
                self.lava_map[lava_indices_x[i], lava_indices_y[i]] = False
    # @profile
    def generate_state_target_tuples(self, max_dist=16):
        if self.DP_info["lava_map"] is None:
            self.init_DP_assets()
        if self.DP_info["P"] is None:
            self.collect_transition_probs()
        if self.DP_info["A"] is None:
            self.collect_state_adjacency()
        if self.DP_info["states_reachable"] is None:
            self.collect_states_reachable()
        goal_i, goal_j = self.pos_monster

        tuples = []
        states_reachable = copy.copy(self.DP_info["states_reachable"])
        ijxds_reachable = np.stack(self.state2ijxd(states_reachable), 1)
        states_reachable_nonterminal = []
        mask_nonterminal_among_reachable = np.zeros(len(states_reachable), dtype=bool)
        for idx_state_reachable in range(len(states_reachable)):
            ijxd = ijxds_reachable[idx_state_reachable]
            i, j = ijxd[0], ijxd[1]
            if self.DP_info["lava_map"][i, j] or i == goal_i and j == goal_j:
                continue  # dont bother if starting from lava or real goal
            else:
                states_reachable_nonterminal.append(states_reachable[idx_state_reachable])
                mask_nonterminal_among_reachable[idx_state_reachable] = True
        A_reduced = self.DP_info["A"][states_reachable_nonterminal, :][:, states_reachable_nonterminal]
        # start_ijds = ijxds_reachable[mask_nonterminal_among_reachable]
        D = floyd_warshall(A_reduced)
        D[D > max_dist] = np.inf
        for ii in range(len(states_reachable_nonterminal)):
            for jj in range(len(states_reachable_nonterminal)):
                if ii == jj or D[ii, jj] >= max_dist:
                    continue
                tuples.append((states_reachable_nonterminal[ii], states_reachable_nonterminal[jj], int(D[ii, jj])))

        self.DP_info["state_target_tuples"] = tuples
        return tuples

    def gen_grid(self, width, height):
        self._gen_grid(width, height)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        self.grid_complete = Grid(width, height)

        # Generate the surrounding walls
        # self.grid.wall_rect(0, 0, width, height)

        self.put_obj(Monster(), self.pos_monster[0], self.pos_monster[1])

        self.grid_complete.set(self.pos_sword[0], self.pos_sword[1], Sword())
        self.grid_complete.set(self.pos_shield[0], self.pos_shield[1], Shield())
        self.grid_complete.set(self.pos_monster[0], self.pos_monster[1], Monster())
        # self.full_grid_base[self.pos_shield[0], self.pos_shield[1], 1] = COLOR_TO_IDX["blue"]
        # self.full_grid_base[self.pos_sword[0], self.pos_sword[1], 1] = COLOR_TO_IDX["purple"]


        for i in range(0, self.lava_map.shape[0]):
            for j in range(0, self.lava_map.shape[1]):
                if self.lava_map[i, j]:
                    self.grid.set(i, j, Lava())
                    self.grid_complete.set(i, j, Lava())

        self.full_grid_base = self.grid.encode(ignore_color=False, ignore_dir=self.ignore_dir)
        self.full_grid_base[self.pos_shield[0], self.pos_shield[1], 1] = COLOR_TO_IDX["blue"]
        self.full_grid_base[self.pos_sword[0], self.pos_sword[1], 1] = COLOR_TO_IDX["purple"]

        # Place the agent
        if self.pos_agent_init is not None:
            self.agent_pos = self.pos_agent_init
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        self.mission = STR_MISSION

    def reset(self, same_init_pos=False):
        super().reset()
        if self.obs_goal is None:
            self.obs_goal = self.draw_obs_with_agent(int(self.pos_monster[0]), int(self.pos_monster[1]), 3, 0, lava_map=None)
        if self.uniform_init and not same_init_pos:
            if self.DP_info["states_reachable"] is None:
                self.collect_states_reachable()
            while True:  # sample a random state in states_reachable and make sure it is not the goal state
                state_rand = int(np.random.choice(self.DP_info["states_reachable"]))
                i, j, x, d = self.state2ijxd(state_rand)
                if not (i == self.pos_monster[0] and j == self.pos_monster[1]) and not self.DP_info["lava_map"][i, j]:
                    break
            self.agent_pos = (int(i), int(j))
            self.sword_acquired, self.shield_acquired = x2swordshield(x) 
            self.agent_dir = int(d)
        else:
            if self.DP_info["states_reachable"] is None:
                self.collect_states_reachable()
            if self.ijxd2state(self.pos_agent_init[0], self.pos_agent_init[1], 0) not in self.DP_info["states_reachable"]:
                raise RuntimeError(f"initial agent position not reachable, [{self.pos_agent_init[0]}, {self.pos_agent_init[1]}]")
            self.agent_pos = copy.copy(self.pos_agent_init)
            self.agent_dir = copy.copy(self.agent_start_dir)
            self.sword_acquired, self.shield_acquired = False, False
        self.obs_curr = self.gen_fullyobservable_obs()
        return self.obs_curr

    def move_forward(self):
        reward, done = 0.0, False
        fwd_pos = self.front_pos
        flag_inside = self.check_inside(fwd_pos)  # check if the tile in front is still inside the boundaries
        if flag_inside:
            fwd_cell = self.grid_complete.get(*fwd_pos) if flag_inside else None
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None:
                if fwd_cell.type == "monster":
                    done = True
                    reward = float(self.sword_acquired and self.shield_acquired)
                elif fwd_cell.type == "lava":
                    done = True
                elif fwd_cell.type == "sword":
                    done = False
                    self.sword_acquired = True
                elif fwd_cell.type == "shield":
                    done = False
                    self.shield_acquired = True
        return reward, done

    def obs2ijxd(self, obs):
        if len(obs.shape) == 3:
            obs = obs[None, :, :, :]
        size_batch, width, height, _ = obs.shape
        slice_type = obs[:, :, :, 0]
        mask_agent = slice_type == OBJECT_TO_IDX["agent"]
        ret_i, ret_j, ret_d = [], [], []
        for idx_sample in range(size_batch):
            found = False
            for i in range(width):
                if found:
                    break
                for j in range(height):
                    if found:
                        break
                    elif mask_agent[idx_sample, i, j]:
                        found = True
                        ret_i.append(i)
                        ret_j.append(j)
                        if not self.ignore_dir:
                            ret_d.append(int(obs[idx_sample, i, j, -1]))
            if not found:
                raise RuntimeError("agent not found in given obs")
        assert len(ret_i) == len(ret_j)
        if not self.ignore_dir:
            assert len(ret_i) == len(ret_d)
        mask_sword = slice_type == OBJECT_TO_IDX["sword"]
        mask_shield = slice_type == OBJECT_TO_IDX["shield"]
        sword_acquired = mask_sword.sum(-2).sum(-1) == 0
        shield_acquired = mask_shield.sum(-2).sum(-1) == 0
        x = swordshield2x(sword_acquired, shield_acquired)
        if len(ret_i) == 1:
            if self.ignore_dir:
                return int(ret_i[0]), int(ret_j[0]), int(x)
            else:
                return int(ret_i[0]), int(ret_j[0]), int(x), int(ret_d[0])
        else:
            agent_i = np.array(ret_i)
            agent_j = np.array(ret_j)
            if self.ignore_dir:
                return agent_i, agent_j, x
            else:
                agent_d = np.array(ret_d)
                return agent_i, agent_j, x, agent_d

    def get_lava_map(self):
        maps = self.full_grid_base[:, :, 0] == OBJECT_TO_IDX["lava"]
        return maps.squeeze()

    @classmethod
    def check_obs_validity(cls, obs):
        if len(obs.shape) == 3:
            obs = obs[None, :, :, :]
        assert len(obs.shape) == 4
        slice_type = obs[:, :, :, 0]
        slice_color = obs[:, :, :, 1]
        mask_agents = slice_type == OBJECT_TO_IDX["agent"]
        num_agents = mask_agents.sum((-1, -2))
        mask_swords = slice_type == OBJECT_TO_IDX["sword"]
        num_swords = mask_swords.sum((-1, -2))
        mask_shields = slice_type == OBJECT_TO_IDX["shield"]
        num_shields = mask_shields.sum((-1, -2))
        mask_monsters = slice_type == OBJECT_TO_IDX["monster"] 
        num_monsters = mask_monsters.sum((-1, -2))
        assert (num_agents == 1).all()
        assert (num_swords <= 1).all() and (num_swords >= 0).all()
        assert (num_shields <= 1).all() and (num_shields >= 0).all()
        assert (num_monsters <= 1).all() and (num_monsters >= 0).all()
        colors_agent = slice_color[mask_agents]
        mask_should_be_red_or_yellow_or_blue_or_purple = num_monsters == 1
        if mask_should_be_red_or_yellow_or_blue_or_purple.any():
            colors_agent_should_be_red_or_yellow_or_blue_or_purple = colors_agent[mask_should_be_red_or_yellow_or_blue_or_purple]
            assert ((colors_agent_should_be_red_or_yellow_or_blue_or_purple == COLOR_TO_IDX["red"]) | (colors_agent_should_be_red_or_yellow_or_blue_or_purple == COLOR_TO_IDX["yellow"]) | (colors_agent_should_be_red_or_yellow_or_blue_or_purple == COLOR_TO_IDX["blue"]) | (colors_agent_should_be_red_or_yellow_or_blue_or_purple == COLOR_TO_IDX["purple"])).all()
        mask_should_be_green = num_monsters == 0 # monster is green, agent turns green if step on monster
        if mask_should_be_green.any():
            colors_agent_should_be_green = colors_agent[mask_should_be_green]
            assert (colors_agent_should_be_green == COLOR_TO_IDX["green"]).all()

    def obs2state(self, obs=None):
        if self.ignore_dir:
            agent_i, agent_j, x = self.obs2ijxd(obs=obs)
            agent_d = np.zeros_like(agent_i)
        else:
            agent_i, agent_j, x, agent_d = self.obs2ijxd(obs=obs)
        return self.ijxd2state(agent_i, agent_j, x, agent_d)

    def obs2ijxdstate(self, obs=None):
        if self.ignore_dir:
            agent_i, agent_j, x = self.obs2ijxd(obs=obs)
            agent_d = np.zeros_like(agent_i)
        else:
            agent_i, agent_j, x, agent_d = self.obs2ijxd(obs=obs)
        return self.ijxd2state(agent_i, agent_j, x, agent_d), (agent_i, agent_j, x, agent_d)

    def generate_oracle(self, pos_monster=None, ijxd_targ=None, include_random=False):
        if ijxd_targ is not None:
            assert pos_monster is None
        self.init_DP_assets()
        r = self.collect_rewards(pos_monster=pos_monster, ijxd_targ=ijxd_targ)
        P = self.collect_transition_probs(pos_monster=pos_monster, ijxd_targ=ijxd_targ)

        Boper_greedy = lambda r, P, v: np.max(r + self.gamma * VmulP(v, P), axis=-1)
        VmulP = lambda v, P: np.matmul(P, v).transpose()
        v0 = np.zeros(self.num_states)
        v_old = v0
        while True:
            v_new = Boper_greedy(r, P, v_old)
            if np.sum(np.abs(v_new - v_old)) <= 1e-6:
                break
            v_old = v_new
        Q_optimal = r + self.gamma * VmulP(v_new, P)

        if include_random:
            Boper_random = lambda r, P, v: np.mean(r + self.gamma * VmulP(v, P), axis=-1)
            v0 = np.zeros(self.num_states)
            v_old = v0
            while True:
                v_new = Boper_random(r, P, v_old)
                if np.sum(np.abs(v_new - v_old)) <= 1e-6:
                    break
                v_old = v_new
            Q_random = r + self.gamma * VmulP(v_new, P)

        goal_i_original, goal_j_original = self.pos_monster
        if pos_monster is None:
            goal_i, goal_j = self.pos_monster
        else:
            goal_i, goal_j = pos_monster
        
        if ijxd_targ is None and pos_monster is None or goal_i == goal_i_original and goal_j == goal_j_original:
            self.DP_info["ijxd_targ"] = np.array([*self.pos_monster, 3])
            self.DP_info["pos_monster"] = self.pos_monster
            self.DP_info["Q_optimal"] = Q_optimal
            self.DP_info["Q_optimal"].flags["WRITEABLE"] = False
            if include_random:
                self.DP_info["Q_random"] = Q_random
                self.DP_info["Q_random"].flags["WRITEABLE"] = False
            return self.DP_info
        else:
            DP_info = {
                "ijxd_targ": np.array(ijxd_targ) if ijxd_targ is not None else np.array([*pos_monster, 3]),
                "pos_monster": pos_monster,
                "num_states": self.DP_info["num_states"],
                "lava_map": self.DP_info["lava_map"],
                "Q_optimal": Q_optimal,
                "r": r,
                "P": P,
            }
            if include_random:
                DP_info["Q_random"] = Q_random
            return DP_info

    # @profile
    def evaluate_action(self, action, obs=None, ijxd_targ=None, DP_info=None):
        if obs is None:
            obs = self.obs_curr
        if DP_info is None:
            DP_info = self.DP_info
        if DP_info["Q_optimal"] is None:
            DP_info = self.generate_oracle(ijxd_targ=ijxd_targ)
        return float(action in self.get_optimal_actions(self.obs2state(obs), DP_info=DP_info))

    def draw_obs_with_agent(self, i, j, x, d, lava_map=None, i_sword=None, j_sword=None, i_shield=None, j_shield=None):
        if lava_map is None:
            if self.DP_info["lava_map"] is None:
                self.init_DP_assets()
            lava_map = self.DP_info["lava_map"]
        if i_sword is None or j_sword is None:
            i_sword, j_sword = self.pos_sword
            i_sword, j_sword = np.full_like(i, i_sword), np.full_like(i, j_sword)
        if i_shield is None or j_shield is None:
            i_shield, j_shield = self.pos_shield
            i_shield, j_shield = np.full_like(i, i_shield), np.full_like(i, j_shield)
        full_grid = np.copy(self.full_grid_base)
        # full_grid[:, :, 1] = 0
        i, j, x, d = np.array(i).reshape(-1, 1), np.array(j).reshape(-1, 1), np.array(x).reshape(-1, 1), np.array(d).reshape(-1, 1)
        i_sword, j_sword, i_shield, j_shield = i_sword.reshape(-1, 1), j_sword.reshape(-1, 1), i_shield.reshape(-1, 1), j_shield.reshape(-1, 1)
        size_batch = i.size
        assert size_batch == j.size == x.size == d.size
        ijds = np.concatenate([i, j, d], 1)
        full_grid = np.repeat(full_grid[np.newaxis, :, :, :], size_batch, axis=0)
        sword_acquired, shield_acquired = x2swordshield(x)
        for idx_sample in range(size_batch):
            _i, _j, _d = ijds[idx_sample].tolist()
            _i_sword, _j_sword = i_sword[idx_sample], j_sword[idx_sample]
            _i_shield, _j_shield = i_shield[idx_sample], j_shield[idx_sample]
            _sword_acquired, _shield_acquired = sword_acquired[idx_sample], shield_acquired[idx_sample]
            if not _sword_acquired:
                if _i != _i_sword or _j != _j_sword:
                    full_grid[idx_sample, _i_sword, _j_sword, 0] = OBJECT_TO_IDX["sword"]
                else:
                    _sword_acquired = True # NOTE(H): in the cases of visualizations, the agents will be planted directly on the map, thus x's could be wrong
            if not _shield_acquired:
                if _i != _i_shield or _j != _j_shield:
                    full_grid[idx_sample, _i_shield, _j_shield, 0] = OBJECT_TO_IDX["shield"]
                else:
                    _shield_acquired = True
            if lava_map[_i, _j]: # color change to yellow if stepped on lava (for full observability)
                full_grid[idx_sample, _i, _j, 1] = COLOR_TO_IDX["yellow"]
            elif _i == self.pos_monster[0] and _j == self.pos_monster[1]: # color change to green if stepped on monster 
                full_grid[idx_sample, _i, _j, 1] = COLOR_TO_IDX["green"]
            elif _i == _i_sword and _j == _j_sword:
                assert _sword_acquired
                full_grid[idx_sample, _i, _j, 1] = COLOR_TO_IDX["purple"] # color change to purple
            elif _i == _i_shield and _j == _j_shield:
                assert _shield_acquired
                full_grid[idx_sample, _i, _j, 1] = COLOR_TO_IDX["blue"] # color change to blue
            full_grid[idx_sample, _i, _j, 0] = OBJECT_TO_IDX["agent"]
            if full_grid[idx_sample, _i, _j, 1] == 0:
                full_grid[idx_sample, _i, _j, 1] = COLOR_TO_IDX["red"] # default color red
            if not self.ignore_dir:
                full_grid[idx_sample, _i, _j, -1] = _d
        if full_grid.shape[0] == 1:
            full_grid = full_grid.squeeze(0)
        return full_grid

    def render_obs(self, obs, highlight=False, tile_size=32):
        return self.get_full_render(highlight, tile_size, obs=obs)

    def render_optimal_policy(self, obs, tile_size=32):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if self.DP_info["Q_optimal"] is None:
            self.generate_oracle()

        width, height = obs.shape[0], obs.shape[1]

        highlight_mask = np.zeros(shape=(width, height), dtype=bool)

        # Compute the total grid size
        width_px = width * tile_size
        height_px = height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        lava_map = (obs[:, :, 0] == OBJECT_TO_IDX["lava"]).squeeze()
        map_agent, agent_dir = obs2agentmap(obs)  # NOTE(H): lots of agents potentially, lol
        if obs[map_agent].squeeze()[1] == COLOR_TO_IDX["yellow"]:
            lava_map[map_agent] = True
        map_sword = obs2swordmap(obs)
        map_shield = obs2shieldmap(obs)
        map_monster = obs2monstermap(obs)
        sword_acquired = map_sword.sum() == 0
        shield_acquired = map_shield.sum() == 0
        x = swordshield2x(sword_acquired, shield_acquired)
        
        for j in range(0, height):
            for i in range(0, width):
                dir_optimal_action = None
                if lava_map[i, j]:
                    cell = Lava()
                elif map_sword[i, j]:
                    cell = Sword()
                elif map_shield[i, j]:
                    cell = Shield()
                elif map_monster[i, j]:
                    cell = Monster()
                else:
                    cell = None
                    state = self.ijxd2state(i, j, x, 0)
                    if state in self.DP_info["states_reachable"]:
                        Q_optimal_cell = self.DP_info["Q_optimal"][state]
                        dir_optimal_action = int(Q_optimal_cell.argmax())

                tile_img = self.grid.render_tile(
                    cell,
                    agent_dir=dir_optimal_action,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img
    
    def render_states_reachable(self, obs, tile_size=32):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if self.DP_info["states_reachable"] is None:
            self.collect_states_reachable()

        width, height = obs.shape[0], obs.shape[1]

        highlight_mask = np.zeros(shape=(width, height), dtype=bool)

        # Compute the total grid size
        width_px = width * tile_size
        height_px = height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        lava_map = (obs[:, :, 0] == OBJECT_TO_IDX["lava"]).squeeze()
        map_agent, agent_dir = obs2agentmap(obs)  # NOTE(H): lots of agents potentially, lol
        if obs[map_agent].squeeze()[1] == COLOR_TO_IDX["yellow"]:
            lava_map[map_agent] = True
        map_sword = obs2swordmap(obs)
        map_shield = obs2shieldmap(obs)
        map_monster = obs2monstermap(obs)
        sword_acquired = map_sword.sum() == 0
        shield_acquired = map_shield.sum() == 0
        x = swordshield2x(sword_acquired, shield_acquired)
        
        for j in range(0, height):
            for i in range(0, width):
                state = self.ijxd2state(i, j, x, 0)
                reachable = state in self.DP_info["states_reachable"]
                if reachable:
                    if lava_map[i, j]:
                        color = "yellow"
                    elif map_sword[i, j]:
                        color = "purple"
                    elif map_shield[i, j]:
                        color = "blue"
                    elif map_monster[i, j]:
                        color = "green"
                    else:
                        color = "grey"
                    cell = Ball(color=color)
                else:
                    if lava_map[i, j]:
                        cell = Lava()
                    elif map_sword[i, j]:
                        cell = Sword()
                    elif map_shield[i, j]:
                        cell = Shield()
                    elif map_monster[i, j]:
                        cell = Monster()
                    else:
                        cell = None
                tile_img = self.grid.render_tile(
                    cell,
                    agent_dir=None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def state2obs(self, state, return_info=False):
        i, j, x, d = self.state2ijxd(state)
        obs = self.draw_obs_with_agent(i, j, x, d)
        SwordShieldMonster.check_obs_validity(obs)
        if return_info:
            return obs, (i, j, x, d)
        else:
            return obs

    def ijxd2obs(self, i, j, x, d=None):
        i, j, x = np.array(i), np.array(j), np.array(x)
        assert i.size == j.size == x.size
        if self.ignore_dir:
            d = np.zeros_like(i)
        else:
            assert d is not None and d.size == i.size
        obs = self.draw_obs_with_agent(i, j, x, d)
        SwordShieldMonster.check_obs_validity(obs)
        return obs

    def collect_rewards(self):
        raise NotImplementedError("implement in subclasses")

    def collect_transition_probs(self):
        raise NotImplementedError("implement in subclasses")

class SwordShieldMonster2(SwordShieldMonster):
    """
    W/ DIRECTIONAL-FORWARD DYNAMICS
    """
    class Actions(IntEnum):
        east = 0  # x+
        south = 1  # y+
        west = 2  # x-
        north = 3  # y-

    def __init__(
        self,
        width=8,
        height=8,
        lava_density_range=[0.3, 0.4],
        gamma=0.99,
        ignore_dir=True,
        uniform_init=False,
        stochasticity=0.0,
        singleton=True,
    ):
        super().__init__(width=width, height=height,
            lava_density_range=lava_density_range,
            gamma=gamma,
            uniform_init=uniform_init,
            stochasticity=stochasticity,
            singleton=singleton,
        )
        self.actions = SwordShieldMonster2.Actions
        self.num_actions = len(self.actions)
        self.action_space = spaces.Discrete(self.num_actions)
        self.gamma = gamma
        self.ignore_dir = bool(ignore_dir)

        self.observation_space = spaces.Box(low=0, high=255, shape=(self.width, self.height, 3 - int(self.ignore_dir)), dtype="uint8")
        self.obs_curr = self.reset()

    def init_DP_assets(self):
        self.num_states = 4 * self.width * self.height
        self.DP_info["num_states"] = self.num_states
        self.DP_info["lava_map"] = self.get_lava_map()

    def collect_rewards(self, pos_sword=None, pos_shield=None, pos_monster=None, lava_map=None, ijxd_targ=None):
        if pos_sword is None and pos_shield is None and pos_monster is None and lava_map is None:
            original_layout = True
            if not self.DP_info["r"] is None and ijxd_targ is None: # original_layout
                return self.DP_info["r"]
        else:
            original_layout = False
        if pos_sword is None:
            pos_sword = self.pos_sword
        if pos_shield is None:
            pos_shield = self.pos_shield
        if pos_monster is None:
            pos_monster = self.pos_monster
        if lava_map is None:
            lava_map = self.DP_info["lava_map"]
        
        if ijxd_targ is None:
            goal_i, goal_j = pos_monster
            r = np.zeros([self.num_states, self.num_actions])
            if goal_j != self.height - 1 and not lava_map[goal_i, goal_j + 1]:
                r[self.ijxd2state(goal_i, goal_j + 1, 3), self.actions.north] = 1
            if goal_i != self.width - 1 and not lava_map[goal_i + 1, goal_j]:
                r[self.ijxd2state(goal_i + 1, goal_j, 3), self.actions.west] = 1
            if goal_j != 0 and not lava_map[goal_i, goal_j - 1]:
                r[self.ijxd2state(goal_i, goal_j - 1, 3), self.actions.south] = 1
            if goal_i != 0 and not lava_map[goal_i - 1, goal_j]:
                r[self.ijxd2state(goal_i - 1, goal_j, 3), self.actions.east] = 1
            if original_layout:
                self.DP_info["r"] = r
                self.DP_info["r"].flags["WRITEABLE"] = False
        else:
            goal_i, goal_j, goal_x = ijxd_targ[0], ijxd_targ[1], ijxd_targ[2]
            r = np.zeros([self.num_states, self.num_actions])
            if goal_j != self.height - 1 and not lava_map[goal_i, goal_j + 1]:
                if goal_i != pos_monster[0] or goal_j + 1 != pos_monster[1]:
                    r[self.ijxd2state(goal_i, goal_j + 1, goal_x), self.actions.north] = 1
            if goal_i != self.width - 1 and not lava_map[goal_i + 1, goal_j]:
                if goal_i + 1 != pos_monster[0] or goal_j != pos_monster[1]:
                    r[self.ijxd2state(goal_i + 1, goal_j, goal_x), self.actions.west] = 1
            if goal_j != 0 and not lava_map[goal_i, goal_j - 1]:
                if goal_i != pos_monster[0] or goal_j - 1 != pos_monster[1]:
                    r[self.ijxd2state(goal_i, goal_j - 1, goal_x), self.actions.south] = 1
            if goal_i != 0 and not lava_map[goal_i - 1, goal_j]:
                if goal_i - 1 != pos_monster[0] or goal_j != pos_monster[1]:
                    r[self.ijxd2state(goal_i - 1, goal_j, goal_x), self.actions.east] = 1
        return r

    def ijxd2state(self, i, j, x, d=None):
        i, j, x = np.array(i, dtype=np.int64), np.array(j, dtype=np.int64), np.array(x, dtype=np.int64)
        if d is not None:
            d = np.array(d, dtype=np.int64)
        assert i.size == j.size == x.size
        if d is not None:
            assert i.size == d.size
        assert (i >= 0).all() and (i < self.width).all()
        assert (j >= 0).all() and (j < self.height).all()
        assert (x >= 0).all() and (x < 4).all()
        return x * self.width * self.height + i * self.width + j

    def state2ijxd(self, state): # validated
        state = np.array(state)
        num_states_per_x = self.width * self.height
        state_local = state % num_states_per_x
        i = state_local // self.width
        j = state_local - i * self.width
        assert i.size == j.size == state.size
        assert (i >= 0).all() and (i < self.width).all()
        assert (j >= 0).all() and (j < self.height).all()
        d = np.zeros_like(state)
        x = state // num_states_per_x
        return i, j, x, d

    def collect_transition_probs(self, pos_sword=None, pos_shield=None, pos_monster=None, lava_map=None, ijxd_targ=None):
        if pos_sword is None and pos_shield is None and pos_monster is None and lava_map is None:
            original_layout = True
            if not self.DP_info["P"] is None and ijxd_targ is None: # original_layout
                return self.DP_info["P"]
        else:
            original_layout = False
        if pos_sword is None:
            pos_sword = self.pos_sword
        if pos_shield is None:
            pos_shield = self.pos_shield
        if pos_monster is None:
            pos_monster = self.pos_monster
        if lava_map is None:
            lava_map = self.DP_info["lava_map"]
        
        if ijxd_targ is not None or self.DP_info["P"] is None:
            if self.DP_info["P"] is None:
                P = np.zeros([self.num_actions, self.num_states, self.num_states], dtype=np.float32)
                for x in range(4):
                    sword_acquired, shield_acquired = x2swordshield(x)
                    for i in range(self.width):
                        for j in range(self.height):
                            if not sword_acquired and pos_sword[0] == i and pos_sword[1] == j:
                                continue
                            if not shield_acquired and pos_shield[0] == i and pos_shield[1] == j:
                                continue
                            idx_state = self.ijxd2state(i, j, x)
                            if (pos_monster[0] == i and pos_monster[1] == j) or lava_map[i, j]:
                                P[:, idx_state, idx_state] = 1.0
                                continue
                            for a in self.actions:
                                dx, dy = DIR_TO_VEC[a]
                                dx, dy = int(dx), int(dy)
                                i_next, j_next = max(0, min(self.width - 1, dx + i)), max(0, min(self.height - 1, dy + j))
                                reaching_sword = pos_sword[0] == i_next and pos_sword[1] == j_next
                                reaching_shield = pos_shield[0] == i_next and pos_shield[1] == j_next
                                assert not (reaching_sword and reaching_shield)
                                x_next = swordshield2x(sword_acquired=reaching_sword or sword_acquired, shield_acquired=reaching_shield or shield_acquired)
                                idx_state_next = self.ijxd2state(i_next, j_next, x_next, 0)
                                P[a, idx_state, idx_state_next] = 1.0
                if original_layout:
                    self.DP_info["P"] = P
                    self.DP_info["P"].flags["WRITEABLE"] = False

            if ijxd_targ is not None:
                P = np.copy(self.DP_info["P"])
                P.flags["WRITEABLE"] = True
                goal_i, goal_j, goal_x = ijxd_targ[0], ijxd_targ[1], ijxd_targ[2]
                idx_state = self.ijxd2state(goal_i, goal_j, goal_x, 0)
                P[:, idx_state, :] = 0.0
                P[:, idx_state, idx_state] = 1.0
                if not self.singleton:
                    if ijxd_targ[0] > 0:
                        idx_state = self.ijxd2state(ijxd_targ[0] - 1, ijxd_targ[1], ijxd_targ[2])
                        P[:, idx_state, :] = 0.0
                        P[:, idx_state, idx_state] = 1.0
                    if ijxd_targ[0] < self.width - 1:
                        idx_state = self.ijxd2state(ijxd_targ[0] + 1, ijxd_targ[1], ijxd_targ[2])
                        P[:, idx_state, :] = 0.0
                        P[:, idx_state, idx_state] = 1.0
                    if ijxd_targ[1] > 0:
                        idx_state = self.ijxd2state(ijxd_targ[0], ijxd_targ[1] - 1, ijxd_targ[2])
                        P[:, idx_state, :] = 0.0
                        P[:, idx_state, idx_state] = 1.0
                    if ijxd_targ[1] < self.height - 1:
                        idx_state = self.ijxd2state(ijxd_targ[0], ijxd_targ[1] + 1, ijxd_targ[2])
                        P[:, idx_state, :] = 0.0
                        P[:, idx_state, idx_state] = 1.0
        return P

    def step(self, action):
        self.step_count += 1
        reward = 0.0
        done, overtime = False, False
        if self.stochasticity > 0:
            if np.random.rand() < self.stochasticity:
                action = np.random.randint(self.num_actions)
        self.agent_dir = action # NOTE(H): this assumes the alignment of directions and actions
        reward, done = self.move_forward()
        if self.step_count >= self.max_steps:
            done, overtime = True, True
        self.agent_dir = 0
        aux = {"overtime": overtime}
        if done:
            aux["pos_monster"] = [self.pos_monster[0], self.pos_monster[1]]
            aux["pos_agent_init"] = self.pos_agent_init
            aux["sword_acquired"] = self.sword_acquired
            aux["shield_acquired"] = self.shield_acquired
        self.obs_curr = self.gen_fullyobservable_obs()
        return self.obs_curr, reward, done, aux
