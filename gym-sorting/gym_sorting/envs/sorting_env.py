import gym
from gym import spaces
from math import inf
import numpy as np
from enum import Enum


class SortingEnv(gym.Env):

    metadata = { 'render.modes': [ 'ascii' ] }

    def __init__(self, init_list):

        self.init_list = init_list

        self.reward_range = (-100, 100)
        self.action_space = spaces.Discrete(n=9)
        self.observation_space = spaces.Discrete(n=4608)

    def reset(self):

        # Data-Specific
        self.i = 0
        self.j = 0
        self.k = 0
        self.len = len(self.init_list)
        self.list = self.init_list

        # RP-Specific
        self.update_flags()
        self.last_action = 0  # NOOP

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

        return int((self.ieq0 << 0)    \
            + (self.jeq0 << 1)         \
            + (self.ieqlen << 2)       \
            + (self.jeqlen << 3)       \
            + (self.keq0 << 4)         \
            + (self.keqlen << 5)       \
            + (self.iltj << 6)         \
            + (self.jlti << 7)         \
            + (self.listigtlistj << 8) \
            + (self.last_action << 9))

    def step(self, action):

        reward = 0
        done = False

        # NOOP
        if action == 0:
            raise Exception('NOOP is not permitted!')

        # TERMINATE
        elif action == 1:

            done = True
            reward = 100 if (self.list == sorted(self.list)) else -100

        # INCI
        elif action == 2:
            self.i = min(self.i + 1, self.len)

        # INCJ
        elif action == 3:
            self.j = min(self.j + 1, self.len)

        # INCK
        elif action == 4:
            self.k = min(self.k + 1, np.iinfo(np.uint64).max)

        # SETIZERO
        elif action == 5:
            self.i = 0
        
        # SETJZERO
        elif action == 6:
            self.j = 0

        # SETKZERO
        elif action == 7:
            self.k = 0

        # SWAP
        elif action == 8:

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
            
        self.last_action = action
        self.update_flags()

        return self.encode_state(), reward, done, {}

    def pretty_last_action(self):

        return [
            'NOOP',
            'TERMINATE',
            'INCI',
            'INCJ',
            'INCK',
            'SETIZERO',
            'SETJZERO',
            'SETKZERO',
            'SWAP'
        ][self.last_action]

    def render(self, mode='ascii'):

        print(f"Data-Specific:")
        print(f"i = {self.i}, j = {self.j}")
        print(f"k = {self.k}, len = {self.len}")
        print(f"list = {self.list}")
        print(f"RP-Specific:")
        print(f"i {'=' if self.ieq0 else '!='} 0, j {'=' if self.jeq0 else '!='} 0")
        print(f"i {'=' if self.ieqlen else '!='} len, j {'=' if self.jeqlen else '!='} len")
        print(f"k {'=' if self.keq0 else '!='} 0, k {'=' if self.keqlen else '!='} len")
        print(f"i {'<' if self.iltj else '>='} j, j {'<' if self.jlti else '>='} i")
        print(f"list[i] {'>' if self.listigtlistj else '<='} list[j]")
        print(self.pretty_last_action())
