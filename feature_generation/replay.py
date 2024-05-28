import numpy as np
import torch

class Replay_generation():
    def __init__(self, size, batch_size):
        self.memory_capacity = size
        self.memory_counter = 0
        self.BATCH_SIZE = batch_size
        self.memory = {}

    def _getBatches(self):
        getBatches_index = np.random.choice(self.memory_capacity, self.BATCH_SIZE)
        return getBatches_index

    def store_transition(self, generation_dict):
        index = self.memory_counter % self.memory_capacity
        self.memory[index] = generation_dict
        self.memory_counter += 1

    def getBatches(self,args,device):
        batch_list = []
        getBatches_index = self._getBatches()
        for i in range(args.episodes):
            states = []
            actions = []
            rewards = []
            states_ = []
            actions_ = []
            for j in getBatches_index:
                state = self.memory[j][i][0]
                action = self.memory[j][i][1]
                reward = self.memory[j][i][2]
                state_ = self.memory[j][i][3]
                action_ = self.memory[j][i][4]
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                states_.append(state_)
                actions_.append(action_)
            states = torch.cat(states,dim=0).detach()
            actions = [torch.tensor(action) for action in actions]
            actions = torch.cat(actions,dim=0).to(device).detach()
            states_ = torch.cat(states_,dim=0).detach()
            actions_ = [torch.tensor(action) for action in actions_]
            actions_ = torch.cat(actions_,dim=0).detach()
            rewards = torch.tensor(rewards, dtype=torch.float)
            reward = rewards.mean().detach()
            batch_list.append((states,actions,reward,states_,actions_))
        return batch_list


class Replay_discrimination():
    def __init__(self, size, batch_size):
        self.memory_capacity = size
        self.memory_counter = 0
        self.batch_size = batch_size
        self.memory = {}

    def store_transition(self, discrimination_dict):
        index = self.memory_counter % self.memory_capacity
        self.memory[index] = discrimination_dict
        self.memory_counter += 1

    def _getBatches(self):
        getBatches_index = np.random.choice(self.memory_capacity, self.batch_size)
        return getBatches_index
    def getBatches(self,args,device):
        batch_list = []
        getBatches_index = self._getBatches()
        for i in range(args.episodes):
            states = []
            actions = []
            rewards = []
            states_ = []
            actions_ = []
            for j in getBatches_index:
                state = self.memory[j][i][0]
                action = self.memory[j][i][1]
                reward = self.memory[j][i][2]
                state_ = self.memory[j][i][3]
                action_ = self.memory[j][i][4]
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                states_.append(state_)
                actions_.append(action_)
            states = torch.cat(states, dim=0).detach()
            actions = [torch.tensor(action) for action in actions]
            actions = torch.cat(actions, dim=0).to(device).detach()
            states_ = torch.cat(states_, dim=0).detach()
            actions_ = [torch.tensor(action) for action in actions_]
            actions_ = torch.cat(actions_, dim=0).detach()
            rewards = torch.tensor(rewards, dtype=torch.float)
            reward = rewards.mean().detach()
            batch_list.append((states, actions, reward, states_, actions_))
        return batch_list
