import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_pos_emb

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Agent1(nn.Module):
    def __init__(self, args, data_nums, feature_nums, operations_c, operations_d, d_model, device):
        super(Agent1, self).__init__()
        self.nums = max(operations_d, operations_c)
        self.args = args
        self.datalayer = nn.Sequential(
            nn.BatchNorm1d(data_nums),
            nn.Linear(data_nums, d_model),
            nn.BatchNorm1d(d_model),
        )
        self.emb = Attention(d_model)
        self.eval_net = QNet_generation(d_model, self.nums)
        self.target_net = QNet_generation(d_model, self.nums)
        self.layernorm = nn.LayerNorm(data_nums)
        self.feature_nums = feature_nums
        self.device = device

    def forward(self, input, for_next, con_or_dis):
        pos_emb = get_pos_emb(input, con_or_dis)
        pos_emb = torch.tensor(pos_emb).unsqueeze(0).to(self.device).to(torch.float32)
        input_norm = self.layernorm(input)
        pos_data = self.datalayer(input_norm).unsqueeze(dim=0)
        pos_data = torch.where(torch.isnan(pos_data), torch.full_like(pos_data, 0), pos_data)
        pos_data = (pos_data + pos_emb)
        emb_data = self.emb(pos_data)
        emb_data = torch.where(torch.isnan(emb_data), torch.full_like(emb_data, 0), emb_data)
        emb_data = emb_data.squeeze(0)
        encoder_mean = self.mean(emb_data, self.feature_nums)
        if for_next:
            generation_logits = self.target_net(encoder_mean)
        else:
            generation_logits = self.eval_net(encoder_mean)
        generation_logits = torch.where(torch.isnan(generation_logits), torch.full_like(generation_logits, 0), generation_logits)
        return generation_logits, encoder_mean

    def mean(self, emb_data, feature_nums):
        num_groups = len(emb_data) // feature_nums
        result = []
        for i in range(feature_nums):
            if i != feature_nums - 1:
                group = emb_data[i * num_groups: (i + 1) * num_groups]
            else:
                group = emb_data[i * num_groups:]
            group_avg = torch.mean(group, dim=0)
            result.append(group_avg)
        result = torch.stack(result)
        return result


class Agent2(nn.Module):
    def __init__(self, args, operations_c, hidden_size, d_model):
        super(Agent2, self).__init__()
        self.args = args
        self.embedding = nn.Embedding(operations_c, hidden_size)
        self.eval_net = QNet_discrimination(hidden_size + d_model, operations_c)
        self.target_net = QNet_discrimination(hidden_size + d_model, operations_c)

    def forward(self, input, emb_generation, for_next):
        embedding_output = self.embedding(input)
        embedding_output = torch.where(torch.isnan(embedding_output), torch.full_like(embedding_output, 0),
                                       embedding_output).squeeze(0)
        all_embedding = torch.cat((embedding_output, emb_generation), dim=1)
        if for_next:
            discrimination_logits = self.target_net(all_embedding)
        else:
            discrimination_logits = self.eval_net(all_embedding)
        discrimination_logits = torch.where(torch.isnan(discrimination_logits), torch.full_like(discrimination_logits, 0), discrimination_logits)
        return discrimination_logits, all_embedding


class QNet_generation(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, init_w=0.1, device=None):
        super(QNet_generation, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.out = nn.Linear(hidden_dim, action_dim)
        self.out.weight.data.normal_(-init_w, init_w)
        self.device = device

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value


class QNet_discrimination(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, init_w=0.1, device=None):
        super(QNet_discrimination, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.out = nn.Linear(hidden_dim, action_dim)
        self.out.weight.data.normal_(-init_w, init_w)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value


class Attention(nn.Module):
    def __init__(self, d_model):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.liner_q = nn.Linear(d_model, 192)
        self.liner_k = nn.Linear(d_model, 192)
        self.liner_v = nn.Linear(d_model, 192)
        self.scale_factor = np.sqrt(32)
        self.softmax = nn.Softmax(dim=-1)
        self.liner3 = nn.Linear(6 * 32, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=d_model, kernel_size=1)
        self.linner1 = nn.Linear(d_model, d_model)
        self.linner2 = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)


    def forward(self, enc_inputs):
        b_size = enc_inputs.shape[0]
        qs = self.liner_q(enc_inputs).view(b_size, -1, 6, 32).transpose(1, 2)
        ks = self.liner_k(enc_inputs).view(b_size, -1, 6, 32).transpose(1, 2)
        vs = self.liner_v(enc_inputs).view(b_size, -1, 6, 32).transpose(1, 2)
        scores = torch.matmul(qs, ks.transpose(-1, -2)) / self.scale_factor
        attn = self.softmax(scores)
        context = torch.matmul(attn, vs)
        context = context.transpose(1, 2).contiguous().view(b_size, -1, 192)
        context = torch.where(torch.isnan(context), torch.full_like(context, 0), context)
        output = self.liner3(context)
        enc_outputs = self.layer_norm(enc_inputs + output)
        output = self.linner1(enc_outputs)
        return self.layer_norm(enc_outputs + output)

