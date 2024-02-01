import torch

gamma = 0.99
batch_size = 32

epsilon = 1.0
lr = 0.0001
initial_exploration = 150
goal_score = 200
log_interval = 10
update_target = 50
replay_memory_capacity = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sequence_length = 8
burn_in_length = 4
grid_num=64

feature_num = 6

human_flag=True


action_space = [3, 4, 5, 6, 7, 8, 9]

#Intervention mode [0: Division 1 only: Division 2 only: Full time intervention]
intervention_mode=0



