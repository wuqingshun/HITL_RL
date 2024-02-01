import random
from collections import namedtuple, deque
from .config_DRQN import sequence_length as sequence_length_DRQN,device as device_DRQN ,grid_num as grid_num_DRQN,feature_num as feature_num_DRQN
from .config_DARQN import sequence_length as sequence_length_DARQN,device as device_DARQN ,grid_num as grid_num_DARQN,feature_num as feature_num_DARQN
from .config_DDAQN import sequence_length as sequence_length_DDAQN,device as device_DDAQN ,grid_num as grid_num_DDAQN,feature_num as feature_num_DDAQN
import numpy as np
import torch

Transition = namedtuple(
    'Transition', ('state', 'next_state', 'action', 'reward', 'human')
)

class Memory_DQN(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, next_state, action, reward, human):
        self.memory.append(Transition(torch.stack(list(state)), torch.stack(list(next_state)), action, reward, human))
    
    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch
    
    def __len__(self):
        return len(self.memory)

class Memory_DRQN(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.local_memory = []
        self.capacity = capacity

    def push(self, state, next_state, action, reward, human):

        self.local_memory.append(Transition(state, next_state, action, reward,human))

        if human == 0:
            while len(self.local_memory) < sequence_length_DRQN:
                self.local_memory.insert(0, Transition(
                    torch.zeros(feature_num_DRQN,grid_num_DRQN,grid_num_DRQN,device=device_DRQN, dtype=torch.float),
                    torch.zeros(feature_num_DRQN,grid_num_DRQN,grid_num_DRQN,device=device_DRQN, dtype=torch.float),
                    [1,0,0,0,0,0,0],
                    0,
                    0,
                ))
            self.memory.append(self.local_memory)
            del self.local_memory
            self.local_memory = []

    def sample(self, batch_size):
        batch_state, batch_next_state, batch_action, batch_reward, batch_human = [], [], [], [], []
        p = np.array([len(episode) for episode in self.memory])
        p = p / p.sum()
        batch_indexes = np.random.choice(np.arange(len(self.memory)), batch_size,p=p)

        for batch_idx in batch_indexes:
            episode = self.memory[batch_idx]

            start = random.randint(0, len(episode) - sequence_length_DRQN)
            transitions = episode[start:start + sequence_length_DRQN]
            batch = Transition(*zip(*transitions))

            # print(batch.state)
            batch_state.append(torch.stack(list(map(lambda s: s.to('cuda'), list(batch.state)))))
            batch_next_state.append(torch.stack(list(map(lambda s: s.to('cuda'), list(batch.next_state)))))
            batch_action.append(torch.Tensor(list(batch.action)))
            batch_reward.append(torch.Tensor(list(batch.reward)))
            batch_human.append(torch.Tensor(list(batch.human)))

        return Transition(batch_state, batch_next_state, batch_action, batch_reward, batch_human)

    def __len__(self):
        return len(self.memory)

class Memory_DARQN(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.local_memory = []
        self.capacity = capacity

    def push(self, state, next_state, action, reward, human):

        self.local_memory.append(Transition(state, next_state, action, reward,human))

        if human == 0:
            while len(self.local_memory) < sequence_length_DARQN:
                self.local_memory.insert(0, Transition(
                    torch.zeros(feature_num_DARQN,grid_num_DARQN,grid_num_DARQN,device=device_DARQN, dtype=torch.float),
                    torch.zeros(feature_num_DARQN,grid_num_DARQN,grid_num_DARQN,device=device_DARQN, dtype=torch.float),
                    [1,0,0,0,0,0,0],
                    0,
                    0,
                ))
            self.memory.append(self.local_memory)
            del self.local_memory
            self.local_memory = []

    def sample(self, batch_size):
        batch_state, batch_next_state, batch_action, batch_reward, batch_human = [], [], [], [], []
        p = np.array([len(episode) for episode in self.memory])
        p = p / p.sum()
        batch_indexes = np.random.choice(np.arange(len(self.memory)), batch_size,p=p)

        for batch_idx in batch_indexes:
            episode = self.memory[batch_idx]

            start = random.randint(0, len(episode) - sequence_length_DRQN)
            transitions = episode[start:start + sequence_length_DRQN]
            batch = Transition(*zip(*transitions))

            # print(batch.state)
            batch_state.append(torch.stack(list(map(lambda s: s.to('cuda'), list(batch.state)))))
            batch_next_state.append(torch.stack(list(map(lambda s: s.to('cuda'), list(batch.next_state)))))
            batch_action.append(torch.Tensor(list(batch.action)))
            batch_reward.append(torch.Tensor(list(batch.reward)))
            batch_human.append(torch.Tensor(list(batch.human)))

        return Transition(batch_state, batch_next_state, batch_action, batch_reward, batch_human)

    def __len__(self):
        return len(self.memory)


class Memory_DDAQN(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.local_memory = []
        self.capacity = capacity
    def push(self, state, next_state, action, reward, human):

        self.local_memory.append(Transition(state, next_state, action, reward,human))

        if human == 0:
            while len(self.local_memory) < sequence_length_DDAQN:
                self.local_memory.insert(0, Transition(
                    torch.zeros(feature_num_DDAQN,grid_num_DDAQN,grid_num_DDAQN,device=device_DDAQN, dtype=torch.float),
                    torch.zeros(feature_num_DDAQN,grid_num_DDAQN,grid_num_DDAQN,device=device_DDAQN, dtype=torch.float),
                    [1,0,0,0,0,0,0],
                    0,
                    0,
                ))
            self.memory.append(self.local_memory)
            del self.local_memory
            self.local_memory = []

    def sample(self, batch_size):
        batch_state, batch_next_state, batch_action, batch_reward, batch_human = [], [], [], [], []
        p = np.array([len(episode) for episode in self.memory])
        p = p / p.sum()
        batch_indexes = np.random.choice(np.arange(len(self.memory)), batch_size,p=p)

        for batch_idx in batch_indexes:
            episode = self.memory[batch_idx]

            start = random.randint(0, len(episode) - sequence_length_DRQN)
            transitions = episode[start:start + sequence_length_DRQN]
            batch = Transition(*zip(*transitions))

            # print(batch.state)
            batch_state.append(torch.stack(list(map(lambda s: s.to('cuda'), list(batch.state)))))
            batch_next_state.append(torch.stack(list(map(lambda s: s.to('cuda'), list(batch.next_state)))))
            batch_action.append(torch.Tensor(list(batch.action)))
            batch_reward.append(torch.Tensor(list(batch.reward)))
            batch_human.append(torch.Tensor(list(batch.human)))

        return Transition(batch_state, batch_next_state, batch_action, batch_reward, batch_human)

    def __len__(self):
        return len(self.memory)

