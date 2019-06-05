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
        self.len = len(self.init_list)

        self.reward_range = (-100, 100)
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *([0] * self.len), 0, 0, 0], dtype=np.uint64), 
                                            high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, self.len, *([self.max] * self.len), self.len, self.len, np.iinfo(np.uint64).max], dtype=np.uint64), 
                                            dtype=np.uint64)

    def reset(self):

        self.list = self.init_list
        self.max = max(self.list)
        self.len = len(self.list)

        self.i = 0
        self.j = 0
        self.k = 0

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
        self.listigtlistj = (self.i < self.len) and (self.j < self.len) and (self.list[self.i] > self.list[self.j])

    def encode_state(self):

        return np.array([
            np.uint64(self.ieq0),
            np.uint64(self.jeq0),
            np.uint64(self.keq0),
            np.uint64(self.iltj),
            np.uint64(self.jlti),
            np.uint64(self.ieqlen),
            np.uint64(self.jeqlen),
            np.uint64(self.keqlen),
            np.uint64(self.listigtlistj),
            np.uint64(self.len),
            *[np.uint64(element) for element in self.list],
            np.uint64(self.i),
            np.uint64(self.j),
            np.uint64(self.k)
        ])

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
