from .const_variable import EnvMode, EnvRule
from .gomoku_env import GomokuEnv

env_rule = EnvRule.BASIC
env_mode = EnvMode.PVP_KEYBOARD
gomoku_env = GomokuEnv(env_rule, env_mode)
gomoku_env.run()
