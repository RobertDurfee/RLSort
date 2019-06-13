import gym
from gym import spaces
from math import inf
import numpy as np
from enum import Enum


class Action(Enum):

    NOOP = 0
    TERMINATE = 1
    INCI = 2
    INCJ = 3
    INCK = 4
    SETIZERO = 5
    SETJZERO = 6
    SETKZERO = 7
    SWAP = 8


class SortingEnv(gym.Env):

    metadata = { 'render.modes': [ 'ascii' ] }

    def __init__(self, init_list):

        self.init_list = init_list

        self.reward_range = (-100, 100)
        self.action_space = spaces.Box(low=0, high=8, dtype=np.byte)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.byte), 
                                            high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 8], dtype=np.byte), 
                                            dtype=np.byte)

    def reset(self):

        # Data-Specific
        self.i = 0
        self.j = 0
        self.k = 0
        self.len = len(self.init_list)
        self.list = self.init_list

        # RP-Specific
        self.update_flags()
        self.last_action = Action.NOOP

        return self.encode_state()

    def update_flags(self):

        self.ieq0 = (self.i == 0)
        self.jeq0 = (self.j == 0)
        self.ieqlen = (self.i == self.len)
        self.jeqlen = (self.j == self.len)
        self.keq0 = (self.k == 0)
        self.keqlen = (self.k == self.len)
        self.iltj = (self.i < self.j)
        self.jlti = (self.j < self.i)
        self.listigtlistj = (self.i < self.len) and (self.j < self.len) and (self.list[self.i] > self.list[self.j])

    def encode_state(self):

        return np.array([
            np.byte(self.ieq0),
            np.byte(self.jeq0),
            np.byte(self.ieqlen),
            np.byte(self.jeqlen),
            np.byte(self.keq0),
            np.byte(self.keqlen),
            np.byte(self.iltj),
            np.byte(self.jlti),
            np.byte(self.listigtlistj),
            np.byte(self.last_action)
        ])

    def step(self, action):

        reward = 0
        done = False

        if Action(action) == Action.NOOP:
            pass
        
        elif Action(action) == Action.TERMINATE:

            done = True
            reward = 100 if (self.list == sorted(self.list)) else -100

        elif Action(action) == Action.INCI:
            self.i = min(self.i + 1, self.len)

        elif Action(action) == Action.INCJ:
            self.j = min(self.j + 1, self.len)
        
        elif Action(action) == Action.INCK:
            self.k = min(self.k + 1, np.iinfo(np.uint64).max)
        
        elif Action(action) == Action.SETIZERO:
            self.i = 0
        
        elif Action(action) == Action.SETJZERO:
            self.j = 0

        elif Action(action) == Action.SETKZERO:
            self.k = 0

        elif Action(action) == Action.SWAP:

            # Out of bounds exception. Swap not possible.
            if (self.i >= self.len) or (self.j >= self.len):
                reward = -10

            # Swap is possible.
            else:

                self.list[self.i], self.list[self.j] = self.list[self.j], self.list[self.i]

                # ..., list_i, ..., list_j, ...
                if self.i < self.j:
                    reward = 10 if (self.list[self.i] < self.list[self.j]) else -10

                # ..., list_j, ..., list_i, ...
                elif self.j < self.i:
                    reward = 10 if (self.list[self.j] < self.list[self.i]) else -10

                # ..., list_i/list_j, ... (swap has no effect)
                else:
                    reward = -10

        self.update_flags()

        return self.encode_state(), reward, done, {}

    def render(self, mode='ascii'):

        print(f"i = {self.i}, j = {self.j}")
        print(f"k = {self.k}, len = {self.len}")
        print(f"list = {self.list}")
        print(f"---------------------------------")
        print(f"i {'=' if self.ieq0 else '!='} 0, j {'=' if self.jeq0 else '!='} 0")
        print(f"i {'=' if self.ieqlen else '!='} len, j {'=' if self.jeqlen else '!='} len")
        print(f"k {'=' if self.keq0 else '!='} 0, k {'=' if self.keqlen else '!='} len")
        print(f"i {'<' if self.iltj else '>='} j, j {'<' if self.jlti else '>='} i")
        print(f"list[i] {'>' if self.listigtlistj else '<='} list[j]")
