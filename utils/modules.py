# import numpy as np
#
# def gather_episodes(env):
#     s, done = env.reset(), False
#
#     for _ in range(5000):
#         a = np.random.choice(env.action_space.n)
#         s_, r, done, info = env.step(a)
#
#         if done:
#
