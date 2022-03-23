from fights.games import gomoku


def main():
    env_rule = gomoku.EnvRule.BASIC
    env_mode = gomoku.EnvMode.PVP_KEYBOARD
    env = gomoku.Env(env_rule, env_mode)
    env.run()


if __name__ == "__main__":
    main()
