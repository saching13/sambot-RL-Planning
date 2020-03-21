from gym.envs.registration import register

register(
    id='adv-diff-field-v0',
    entry_point='gym_adv_diff_field.envs:AdvectionDiffusionFieldEnv'
)