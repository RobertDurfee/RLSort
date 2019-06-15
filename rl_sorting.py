import itertools
import numpy as np
import gym


def generate_training_set(length):

    training_set = list(itertools.permutations(range(length)))
    training_set = list(reversed(training_set))
    training_set = [list(training_list) for training_list in training_set]

    return training_set


def train(length, epochs, batch_size, termination_bound, epsilon, gamma):

    # Initialize training set
    training_set = generate_training_set(length)
    training_slice = training_set[1:]

    # Initialize persistent tables
    q_table = {}
    r_table = {}
    s_table = {}
    visit_table = {}

    at_least_one_fails = True
    epoch = 0

    # While there is a list we cannot sort and more iterations remain
    while at_least_one_fails and epoch < epochs:

        print(f"Epoch {epoch + 1}: ", end='')

        batch = 0

        # While more training epochs remain
        while batch < batch_size:

            # Train on every list in the training slice
            for training_list in training_slice:

                # Create list sorting environment
                env = gym.make('gym_sorting:sorting-v0', init_list=training_list)

                # Perform the q-learning
                learn(env, q_table, r_table, s_table, visit_table, epsilon, gamma, termination_bound)

            batch += 1
        
        at_least_one_fails = False

        # Evaluate on every list in the training slice first
        for training_list in training_slice:

            # Create list sorting environment
            env = gym.make('gym_sorting:sorting-v0', init_list=training_list)

            # Execute the best policy given the q-table
            s = execute(env, q_table, termination_bound)

            # Fail if the list is not sorted
            if env.list != sorted(env.list):

               print('Failed to sort in training slice.')

               at_least_one_fails = True 

               # Don't continue evaluation if one fails!
               break

            # Fail if the last action was not TERMINATE
            elif ((s >> 9) & 15) != 1:

               print('Failed to terminate in training slice.')

               at_least_one_fails = True 

               # Don't continue evaluation if one fails!
               break
        
        # If we sorted every list in the training slice, check the whole training set
        if not at_least_one_fails:

            # For each list in the complete training set
            for training_list in training_slice:

                # Create a sorting environment
                env = gym.make('gym_sorting:sorting-v0', init_list=training_list)

                # Execute the best policy given the q-table
                s = execute(env, q_table, termination_bound)

                # Fail if the list is not sorted
                if env.list != sorted(env.list):

                    print('Failed to sort in complete training set.')

                    at_least_one_fails = True

                    # Add the failed list to the training slice
                    training_slice += [training_list]

                    # Don't continue evaluation if one fails!
                    break

                # Fail if the last action was not TERMINATE
                elif ((s >> 9) & 15) != 1:

                    print('Failed to terminate in complete training set.')

                    at_least_one_fails = True

                    # Add the failed list to the training slice
                    training_slice += [training_list]

                    # Don't continue evaluation if one fails!
                    break
        
        epoch += 1

    return q_table


def learn(env, q_table, r_table, s_table, visit_table, epsilon, gamma, max_iterations, render=False):

    # Make sure the environment is initialized
    s = env.reset()
    if render:
        env.render()

    iteration = 0

    # While last action is not TERMINATE and more iterations remain
    while ((s >> 9) & 15 != 1) and (iteration < max_iterations):

        # Choose action randomly if last action NOOP or all actions have negative value or in exploration mode
        if ((s >> 9) & 15 == 0) or (max(q_table.get(s, {}).values(), default=-np.inf) < 0) or (np.random.random() < epsilon):
            a = np.random.choice(range(1, env.action_space.n))
        
        # Choose action greedily
        else:
            a = max(q_table[s].keys(), key=(lambda key: q_table[s][key]))

        # Perform action
        s_prime, r, done, _ = env.step(a)
        if render:
            env.render()

        # Update tables for P(r | s, a) and P(s' | s, a) calculation
        r_table.setdefault((s, a), {})[r] = r_table.get((s, a), {}).get(r, 0) + 1
        s_table.setdefault((s, a), {})[s_prime] = s_table.get((s, a), {}).get(s_prime, 0) + 1

        # Increase number of visits to (s, a)
        visit_table.setdefault(s, {})[a] = visit_table.get(s, {}).get(a, 0) + 1

        # Calculate decay factor alpha
        alpha = 1 / visit_table[s][a]

        # Update Q(s, a) = (1 - alpha) Q(s, a) + alpha * (sum_i P(r_i | s, a) * r_i + gamma * sum_i P(s_i | s, a) * max_a' Q(s_i, a'))
        q_table.setdefault(s, {})[a] = (1 - alpha) * q_table.get(s, {}).get(a, 0) + alpha * (sum([(r_freq / visit_table[s][a]) * r for r, r_freq in r_table[(s, a)].items()]) \
            + gamma * sum([(s_prime_freq / visit_table[s][a]) * max(q_table.get(s_prime, {}).values(), default=0) for s_prime, s_prime_freq in s_table[(s, a)].items()]))
        
        s = s_prime
        iteration += 1


def execute(env, q_table, termination_threshold, render=False):

    # Make sure environment is initialized
    s = env.reset()
    if render:
        env.render()

    iteration = 0

    # While last action is not TERMINATE and more iterations remain
    while (((s >> 9) & 15) != 1) and (iteration < termination_threshold):

        # Choose a = argmax_a' Q(s, a')
        a = max(q_table[s].keys(), key=(lambda key: q_table[s][key]))

        # Perform action
        s, _, _, _ = env.step(a)
        if render:
            env.render()

        iteration += 1
    
    return s
