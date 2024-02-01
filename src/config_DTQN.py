import torch

env_name = 'CartPole-v1'
gamma = 0.99
batch_size = 32
lr = 0.001
initial_exploration = 100
goal_score = 200
log_interval = 10
update_target = 50
replay_memory_capacity = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sequence_length = 8
burn_in_length = 4


action_space = [3, 4, 5, 6, 7, 8, 9]