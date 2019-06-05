import gym
from gym import spaces
from math import inf
import numpy as np

class SortingEnv(gym.Env):

    metadata = { 'render.modes': [ 'human' ] }

    # init_list can only contain non-negative integers!!!
    def __init__(self, init_list):

        self.init_list = init_list
        self.max = max(self.init_list)

        self.reward_range = (-100, 100)
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=0, high=np.iinfo(np.uint64).max, shape=(1,), dtype=np.uint64)

    def reset(self):

        self.i = 0
        self.j = 0
        self.k = 0
        self.len = len(self.init_list)
        self.list = self.init_list

        self.update_flags()

        return self.encode_state()

    def update_flags(self):

        self.ieq0 = (self.i == 0)
        self.jeq0 = (self.j == 0)
        self.keq0 = (self.k == 0)
        self.iltj = (self.i < self.j)
        self.jlti = (self.j < self.i)
        self.ieqlen = (self.i == self.len)
        self.jeqlen = (self.j == self.len)
        self.keqlen = (self.k == self.len)
        self.listigtlistj = (self.list[self.i] > self.list[self.j])

    def encode_state(self):

        val_bits_pairs = [
            (self.ieq0,         int(1).bit_length()),
            (self.jeq0,         int(1).bit_length()),
            (self.keq0,         int(1).bit_length()),
            (self.iltj,         int(1).bit_length()),
            (self.jlti,         int(1).bit_length()),
            (self.ieqlen,       int(1).bit_length()),
            (self.jeqlen,       int(1).bit_length()),
            (self.keqlen,       int(1).bit_length()),
            (self.listigtlistj, int(1).bit_length()),
            (self.len,          self.len.bit_length()),
            *zip(self.list,     [max(self.list).bit_length()] * self.len),
            (self.i,            self.len.bit_length()),
            (self.j,            self.len.bit_length()),
            (self.k,            0)  # This has an infinite bound. Zero is a placeholder.
        ]

        state, shift = 0, 0

        for val, bits in val_bits_pairs:

            state += (val << shift)
            shift += bits

        return np.array([state], dtype=np.uint64)

    def step(self, action):

        reward = 0
        done = False

        # NOOP
        if action == 0:
            pass
        
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
            self.k += 1
        
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

        self.update_flags()

        return self.encode_state(), reward, done, {}

    def render(self, mode='human'):

        print(f"i = {self.i}, j = {self.j}")
        print(f"k = {self.k}, len = {self.len}")
        print(f"list = {self.list}")
        print(f"---------------------------------")
        print(f"i {'=' if self.ieq0 else '!='} 0, j {'=' if self.jeq0 else '!='} 0")
        print(f"i {'=' if self.ieqlen else '!='} len, j {'=' if self.jeqlen else '!='} len")
        print(f"k {'=' if self.keq0 else '!='} 0, k {'=' if self.keqlen else '!='} len")
        print(f"i {'<' if self.iltj else '>='} j, j {'<' if self.jlti else '>='} i")
        print(f"list[i] {'>' if self.listigtlistj else '<='} list[j]")
