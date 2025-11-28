from gymnasium.envs.registration import register
from .env import CublinoContraEnv

register(
    id="CublinoContra-v0",
    entry_point="src.cublino_contra.env:CublinoContraEnv",
)
