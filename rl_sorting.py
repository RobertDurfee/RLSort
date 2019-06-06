import itertools
import gym

def episodic_training(list_len=4, subset_size=1):

    T = list(itertools.permutations(range(list_len)))
    X = T[:-subset_size:-1]

    at_least_one_list_fails = True
    iteration = 0

    while at_least_one_list_fails and (iteration < 100_000):

        not_maxed_out = True

        while not_maxed_out:

            iteration = iteration + 1
            not_maxed_out = False

            for L in X:
            
                env = gym.make('gym_sorting:sorting-v0', init_list=L)
                env.reset()

def delayed_q_learning():
    pass
