from gym.envs.registration import register

register(
    id='sorting-v0',
    entry_point='gym_sorting.envs:SortingEnv'
)
