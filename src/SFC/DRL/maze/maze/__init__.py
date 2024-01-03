from gym.envs.registration import register

register(
    id="maze-v0",
    entry_point="src.SFC.DRL.maze.maze.envs:SfcEnvironment",
)

register(
    id='maze-v1',
    entry_point='src.SFC.DRL.maze.maze.envs:SfcNoGcn',
)