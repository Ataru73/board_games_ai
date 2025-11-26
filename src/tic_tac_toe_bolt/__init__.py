from gymnasium.envs.registration import register

register(
    id="TicTacToeBolt-v0",
    entry_point="tic_tac_toe_bolt.env:TicTacToeBoltEnv",
)
