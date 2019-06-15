import itertools
import numpy as np
import gym


def generate_training_set(length):

    training_set = list(itertools.permutations(range(length)))
    training_set = list(reversed(training_set))
    training_set = [list(training_list) for training_list in training_set]

    return training_set


def training(length, epochs):

    training_set = generate_training_set(length)
    training_slice = training_set[1:]

    q_table = {}
    r_table = {}
    s_table = {}

    at_least_one_fails = True
    epoch = 0

    while at_least_one_fails and epoch < epochs:

        while epoch < epochs:

            for training_list in training_slice:

                env = gym.make('gym_sorting:sorting-v0', init_list=training_list)
                learn(env, q_table, r_table, s_table)

            epoch += 1
        
        at_least_one_fails = False

        for training_list in training_slice:

            resulting_list = execute(training_list, q_table)

            if (resulting_list is None) or (resulting_list != sorted(resulting_list)):

               at_least_one_fails = True 
               break
        
        if not at_least_one_fails:

            for training_list in training_slice:

                resulting_list = execute(training_list, q_table)

                if (resulting_list is None) or (resulting_list != sorted(resulting_list)):

                    at_least_one_fails = True
                    training_slice += [training_list]
                    break
    
    if epoch >= epochs:
        return None

    return policy()


def learn(env, q_table, r_table, s_table, epsilon, alpha, gamma, max_iterations):

    s = env.reset()
    iteration = 0

    while ((s >> 9) & 15 != 1) and (iteration < max_iterations):

        # Choose action randomly
        if ((s >> 9) & 15 == 0) or (max(q_table[s].values(), default=-np.inf) < 0) or (np.random.random() < epsilon):
            a = np.random.choice(range(1, env.action_space.n))
        
        # Choose action greedily
        else:
            a = max(q_table[s].keys(), key=(lambda key: q_table[s][key]))

        s_prime, r, done, _ = env.step(a)

        r_table.setdefault((s, a), {})[r] = r_table.get((s, a), {}).get(r, 0) + 1
        s_table.setdefault((s, a), {})[s_prime] = s_table.get((s, a), {}).get(s_prime, 0) + 1

        total_r_freq = sum(r_table[(s, a)].values())
        total_s_prime_freq = sum(s_table[(s, a)].values())

        q_table[s][a] = (1 - alpha) * q_table[(s, a)] + alpha * (sum([(r_freq / total_r_freq) * r for r, r_freq in r_table[(s, a)].items()]) \
            + gamma * sum([(s_prime_freq / total_s_prime_freq) * max(q_table[s_prime].values(), default=0) for s_prime, s_prime_freq in s_table[(s, a)].items()]))
        
        s = s_prime
        iteration += 1


def execute():
    pass


def policy():
    pass
