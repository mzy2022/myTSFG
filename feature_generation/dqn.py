import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .embedding_policy import Agent1, Agent2


class DQN_generation(object):
    def __init__(self, args, data_nums, feature_nums,operations_c, operations_d, d_model,
                 memory,device,EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200):
        self.args = args
        self.epochs = args.epochs
        self.episodes = args.episodes
        self.operations_c = operations_c
        self.operations_d = operations_d
        self.device = device
        self.memory = memory
        self.generation_c_num = operations_c
        self.generation_d_num = operations_d
        self.eps_end = EPS_END
        self.eps_start = EPS_START
        self.eps_decay = EPS_DECAY
        self.TARGET_REPLACE_ITER = 50
        self.batch_size = 8
        self.gamma = 0.99
        self.loss_func = nn.MSELoss()
        self.learn_step_counter = 0
        self.agent1 = Agent1(args, data_nums, feature_nums,operations_c, operations_d,d_model,self.device).to(self.device)
        self.agent1_opt = optim.Adam(params=self.agent1.parameters(), lr=args.lr)
        self.nums = max(self.operations_c,self.operations_d)


    def choose_action_generation(self, input, for_next,steps_done,con_or_dis):
        actions_generation = []
        self.agent1.train()
        generation_vals, state = self.agent1(input.to(self.device), for_next,con_or_dis)
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.0 * steps_done / self.eps_decay)
        generation_vals = generation_vals.detach()
        for index, out in enumerate(generation_vals):
            if for_next:
                act = torch.argmax(out).item()
            else:
                if np.random.uniform() > eps_threshold:
                    act = torch.argmax(out)
                else:
                    act = np.random.randint(0, self.nums)
            actions_generation.append(int(act))

        return actions_generation, state


    def store_transition(self,args,batchs):
        store_generation_list = []
        for num, batch in enumerate(batchs):
            states_generation = batch.states_generation
            generation = batch.actions_generation
            reward = batch.reward_1
            states_generation_ = batch.states_generation_
            generation_ = batch.actions_generation_
            store_generation_list.append([states_generation,generation,reward,states_generation_,generation_])
        self.memory.store_transition(store_generation_list)



    def learn(self, args, batchs,device):
        if self.memory.memory_counter >= self.memory.memory_capacity:
            if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
                self.agent1.target_net.load_state_dict(self.agent1.eval_net.state_dict())
            self.learn_step_counter += 1
            loss = 0
            batch_lists = self.memory.getBatches(args,device)

            for memory_list in batch_lists:
                q_next = self.agent1.target_net(memory_list[3])
                q_target = memory_list[2] + self.gamma * torch.max(q_next, dim=-1)[0]
                q_eval = self.agent1.eval_net(memory_list[0])
                actions = torch.tensor(memory_list[1],dtype=torch.long).unsqueeze(0)
                q_eval_selected = q_eval.gather(1, actions).squeeze(0)
                loss = loss + self.loss_func(q_eval_selected, q_target)

            loss /= len(batch_lists)
            self.agent1_opt.zero_grad()
            loss.backward()
            self.agent1_opt.step()


class DQN_discrimination(object):
    def __init__(self, args,operations_c,operations_d,hidden_size,d_model,memory,device,EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200):
        self.args = args
        self.epochs = args.epochs
        self.episodes = args.episodes
        self.device = device
        self.memory = memory
        self.discrimination_nums = max(operations_c,operations_d)
        self.eps_end =EPS_END
        self.eps_start = EPS_START
        self.eps_decay = EPS_DECAY
        self.loss_func = nn.MSELoss()
        self.learn_step_counter = 0
        self.TARGET_REPLACE_ITER = 50
        self.gamma = 0.99
        self.batch_size = 8
        self.agent2 = Agent2(args, self.discrimination_nums,hidden_size,d_model).to(self.device)
        self.agent2_opt = optim.Adam(params=self.agent2.parameters(), lr=args.lr)


    def choose_action_discrimination(self, input, states_generation,for_next,steps_done):
        input = torch.tensor(input)
        actions_discrimination = []
        self.agent2.train()
        discrimination_vals, state = self.agent2(input.to(self.device),states_generation, for_next)
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.0 * steps_done / self.eps_decay)
        discrimination_vals = discrimination_vals.detach()
        for index, out in enumerate(discrimination_vals):
            if for_next:
                act = torch.argmax(out).item()
            else:
                if np.random.uniform() > eps_threshold:
                    act = torch.argmax(out)
                else:
                    act = np.random.randint(0, self.discrimination_nums)
            actions_discrimination.append(int(act))

        return actions_discrimination, state

    def store_transition(self,args,batchs):
        store_discrimination_list = []
        for num, batch in enumerate(batchs):
            states_discrimination = batch.states_discrimination
            discrimination = batch.actions_discrimination
            reward = batch.reward_2
            states_discrimination_ = batch.states_discrimination_
            discrimination_ = batch.actions_discrimination_
            store_discrimination_list.append([states_discrimination, discrimination, reward, states_discrimination_, discrimination_])
        self.memory.store_transition(store_discrimination_list)


    def learn(self, args, batchs,device):
        if self.memory.memory_counter >= self.memory.memory_capacity:
            if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
                self.agent2.target_net.load_state_dict(self.agent2.eval_net.state_dict())
            self.learn_step_counter += 1
            loss = 0
            batch_lists = self.memory.getBatches(args, device)
            for memory_list in batch_lists:
                q_next = self.agent2.target_net(memory_list[3])
                q_target = memory_list[2] + self.gamma * torch.max(q_next, dim=-1)[0]
                q_eval = self.agent2.eval_net(memory_list[0])
                actions = torch.tensor(memory_list[1]).unsqueeze(0)
                q_eval = q_eval.gather(1, actions).squeeze(0)
                loss = loss + self.loss_func(q_eval, q_target)

            loss /= len(batch_lists)
            self.agent2_opt.zero_grad()
            loss.backward()
            self.agent2_opt.step()


