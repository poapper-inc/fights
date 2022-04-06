from fights.games import gomoku


def main():
    env_rule = gomoku.EnvRule.BASIC
    env_mode = gomoku.EnvMode.PVP_KEYBOARD
    env = gomoku.env(env_rule, env_mode)
    env.run()

    env = gomoku.env()
    env.reset()

    # Reference
    # env = knights_archers_zombies_v9.env()
    # env.reset()
    # for agent in env.agent_iter():
    #     observation, reward, done, info = env.last()
    #     action = policy(observation, agent)
    #     env.step(action)

    # env = gym.make("CartPole-v1")
    # observation = env.reset()
    # for _ in range(1000):
    #     env.render()
    #     action = env.action_space.sample()  # your agent here (this takes random actions)
    #     observation, reward, done, info = env.step(action)
    #
    #     if done:
    #         observation = env.reset()
    # env.close()


if __name__ == "__main__":
    main()
