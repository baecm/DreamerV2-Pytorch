### HYPERPARAMETERS ###
model_path = 'result/model/'
image_path = 'result/image/'

env_name = "Pong-v0"

batch = 64
L = 50 #seq len world training
history_size = 64
lr_world = 2e-4

H = 15 #imagination length
gamma = 0.995 #discount factor
lamb = 0.95 #lambda-target
lr_actor = 4e-5
lr_critic = 1e-4
target_interval = 100 #update interval for target critic

gradient_clipping = 100
adam_eps = 1e-5
decay = 1e-6
