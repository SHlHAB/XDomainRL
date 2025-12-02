import gymnasium as gym
import torch
import random
import torch.nn as nn
import collections
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Normal
from itertools import cycle
import argparse
from tqdm import trange
import os
import warnings
import math
from itertools import combinations
from torch.optim.lr_scheduler import StepLR
import torch.nn.init as init
import time
import wandb
# SAC credit to https://github.com/quantumiracle/Popular-RL-Algorithms/tree/master

warnings.filterwarnings('ignore')
# wandb.init(project="cdpc", name = 'reacher')


def compute_source_R(state, action):
    vec = state[-3:]#.detach().cpu().numpy()    # last 3 dim
    reward_dist = -torch.norm(vec)
    reward_ctrl = -np.square(action).sum()
    reward = reward_dist + reward_ctrl 
    return reward 

def compute_target_R(state, action):
    vec = state[-3:]#.detach().cpu().numpy()    # last 3 dim
    reward_dist = -np.linalg.norm(vec)
    reward_ctrl = -np.square(action).sum()
    reward = reward_dist + reward_ctrl 
    return reward 

class ReplayBuffer():
    def __init__(self, buffer_maxlen):
        self.buffer = collections.deque(maxlen=buffer_maxlen)

    def push(self, data):
        self.buffer.append(data)

    def sample(self, batch_size):
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        done_list = []

        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            s, a, r, n_s, d = experience
            # state, action, reward, next_state, done

            state_list.append(s)
            action_list.append(a)
            reward_list.append(r)
            next_state_list.append(n_s)
            done_list.append(d)

        return torch.FloatTensor(state_list).to(device), \
               torch.FloatTensor(action_list).to(device), \
               torch.FloatTensor(reward_list).unsqueeze(-1).to(device), \
               torch.FloatTensor(next_state_list).to(device), \
               torch.FloatTensor(done_list).unsqueeze(-1).to(device)

    def buffer_len(self):
        return len(self.buffer)
    

class ReplayBuffer_target():
    def __init__(self):
        self.buffer = []

    def push(self, total_rewards, count, dist, state, next_state, action):
        trajectory_info = {
        'total_rewards': total_rewards,
        'count': count,
        'dist': dist,
        'state': state,
        'next_state': next_state,
        'action': action
        }
        self.buffer.append(trajectory_info)

    def sample(self, combo):
        trajectory_a = self.buffer[combo[0]]
        trajectory_b = self.buffer[combo[1]]

        return trajectory_a, trajectory_b

    def buffer_len(self):
        return len(self.buffer)


# Value Net
class ValueNet(nn.Module):
    def __init__(self, state_dim, edge=3e-3):
        super(ValueNet, self).__init__()
        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, 1)

        self.linear4.weight.data.uniform_(-edge, edge)
        self.linear4.bias.data.uniform_(-edge, edge)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)

        return x


# Soft Q Net
class SoftQNet(nn.Module):
    def __init__(self, state_dim, action_dim, edge=3e-3):
        super(SoftQNet, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 1)

        self.linear4.weight.data.uniform_(-edge, edge)
        self.linear4.bias.data.uniform_(-edge, edge)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)

        return x


# Policy Net
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2, edge=3e-3):
        super(PolicyNet, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(state_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 512)

        self.mean_linear = nn.Linear(512, action_dim)
        self.mean_linear.weight.data.uniform_(-edge, edge)
        self.mean_linear.bias.data.uniform_(-edge, edge)

        self.log_std_linear = nn.Linear(512, action_dim)
        self.log_std_linear.weight.data.uniform_(-edge, edge)
        self.log_std_linear.bias.data.uniform_(-edge, edge)

        self.action_range = 1.0

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def action(self, state):
        state = torch.FloatTensor(state).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(0, 1)
        z = normal.sample()
        action = self.action_range * torch.tanh(mean + std*z).detach().cpu().numpy()  #z # mean
        return action
    
    def action_val(self, state):
        state = torch.FloatTensor(state).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(0, 1)
        z = normal.sample()
        action = self.action_range * torch.tanh(mean).detach().cpu().numpy()  #z # mean
        return action

    # Use re-parameterization tick
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        noise = Normal(0, 1)

        z = noise.sample()
        action_0 = torch.tanh(mean + std * z.to(device))
        action = action_0 * self.action_range
        log_prob = normal.log_prob(mean + std * z.to(device)) - torch.log(1 - action_0.pow(2) + epsilon) - np.log(self.action_range)
       
        return action, log_prob


class SAC:
    def __init__(self, env, gamma, tau, buffer_maxlen, q_lr, policy_lr, alpha_lr):

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_lr = 0.001
        self.lr = 0.001
        

        # hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = 128

        # initialize networks
        self.q1_net = SoftQNet(self.state_dim, self.action_dim).to(device)
        self.q2_net = SoftQNet(self.state_dim, self.action_dim).to(device)
        self.q1_net_target = SoftQNet(self.state_dim, self.action_dim).to(device)
        self.q2_net_target = SoftQNet(self.state_dim, self.action_dim).to(device)
        self.policy_net = PolicyNet(self.state_dim, self.action_dim).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device = args.device)#.to(device)

        # Load the target value network parameters
        for target_param, param in zip(self.q1_net_target.parameters(), self.q1_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.q2_net_target.parameters(), self.q2_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Initialize the optimizer
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=q_lr)
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        # Initialize thebuffer
        self.buffer = ReplayBuffer(buffer_maxlen)

        
    def get_action(self, state):
        action = self.policy_net.action(state)

        return action
    def get_action_val(self, state):
        action = self.policy_net.action_val(state)

        return action

        
    def update(self, batch_size):
        state, action, reward, next_state, done = self.buffer.sample(batch_size)

        # soft q loss
        reward_scale = 10
        new_action, log_prob = self.policy_net.evaluate(state)
        new_next_action, log_next_prob = self.policy_net.evaluate(next_state)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)

        # training alpha
        target_entropy = -2
        alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        
        # training soft q
        q1_value = self.q1_net(state, action)
        q2_value = self.q2_net(state, action)
        target_value = torch.min(self.q1_net_target(next_state, new_next_action), self.q2_net_target(next_state, new_next_action))
        target_q_value = reward + (1-done) * self.gamma * target_value
        q1_value_loss = F.mse_loss(q1_value, target_q_value.detach())
        q2_value_loss = F.mse_loss(q2_value, target_q_value.detach())
        
        self.q1_optimizer.zero_grad()
        q1_value_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_value_loss.backward()
        self.q2_optimizer.step()

        # training policy
        predicted_new_q_value = torch.min(self.q1_net(state, new_action),self.q2_net(state, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        
        policy_loss.backward()
        self.policy_optimizer.step()


        # Update target networks
        for target_param, param in zip(self.q1_net_target.parameters(), self.q1_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.q2_net_target.parameters(), self.q2_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        

    
    
class SAC_target:
    def __init__(self, env, gamma, tau, buffer_maxlen, q_lr, policy_lr, alpha_lr):

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]


        # hyperparameters
        self.gamma = gamma
        self.tau = tau

        # initialize networks
        self.q1_net = SoftQNet(self.state_dim, self.action_dim).to(device)
        self.q2_net = SoftQNet(self.state_dim, self.action_dim).to(device)
        self.q1_net_target = SoftQNet(self.state_dim, self.action_dim).to(device)
        self.q2_net_target = SoftQNet(self.state_dim, self.action_dim).to(device)
        self.policy_net = PolicyNet(self.state_dim, self.action_dim).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device = args.device)#.to(device)

        # Load the target value network parameters
        for target_param, param in zip(self.q1_net_target.parameters(), self.q1_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.q2_net_target.parameters(), self.q2_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Initialize the optimizer
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=q_lr)
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        # Initialize thebuffer
        self.buffer = ReplayBuffer(buffer_maxlen)
        self.buffer_ = ReplayBuffer(buffer_maxlen)

        # initialize policy network parameters
        self.mpc_policy_net = MPC_Policy_Net(self.state_dim, self.action_dim).to(device)
        self.mpc_policy_optimizer = optim.Adam(self.mpc_policy_net.parameters(), lr=1e-3, weight_decay=1e-4)

        # initialize dynamic model parameters
        self.dynamic_model = Dynamic_Model(self.state_dim+self.action_dim, self.state_dim).to(device)
        self.dynamic_model_optimizer = optim.Adam(self.dynamic_model.parameters(), lr=1e-3, weight_decay=1e-4)


    def get_action(self, state):
        action = self.policy_net.action(state)

        return action

    def get_action_val(self, state):
        action = self.policy_net.action_val(state)

        return action

        
    def update(self, batch_size):
        # update SAC
        state, action, reward, next_state, done = self.buffer.sample(batch_size)

        
        # soft q loss
        reward_scale = 10
        new_action, log_prob = self.policy_net.evaluate(state)
        new_next_action, log_next_prob = self.policy_net.evaluate(next_state)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)

        # training alpha
        target_entropy = -2
        alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        

        # training soft q
        q1_value = self.q1_net(state, action)
        q2_value = self.q2_net(state, action)
        target_value = torch.min(self.q1_net_target(next_state, new_next_action), self.q2_net_target(next_state, new_next_action))
        target_q_value = reward + (1-done) * self.gamma * target_value
        q1_value_loss = F.mse_loss(q1_value, target_q_value.detach())
        q2_value_loss = F.mse_loss(q2_value, target_q_value.detach())

        self.q1_optimizer.zero_grad()
        q1_value_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_value_loss.backward()
        self.q2_optimizer.step()

        # training policy
        predicted_new_q_value = torch.min(self.q1_net(state, new_action),self.q2_net(state, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()


        # Update target networks
        for target_param, param in zip(self.q1_net_target.parameters(), self.q1_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.q2_net_target.parameters(), self.q2_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        

    
    def update_(self, batch_size):
        # update MPC policy net
        state, action, reward, next_state, done = self.buffer_.sample(batch_size)
        pred_action = self.mpc_policy_net(state)
        loss_mpc = F.mse_loss(pred_action, action)
        self.mpc_policy_optimizer.zero_grad()
        loss_mpc.backward()
        self.mpc_policy_optimizer.step()

        # update dynamic model
        state, action, reward, next_state, done = self.buffer_.sample(batch_size)
        pred_next_state = self.dynamic_model(torch.cat([state, action], dim=1))
        # pred_next_state = self.dynamic_model(state, action)
        loss_dm = F.mse_loss(pred_next_state, next_state)
        self.dynamic_model_optimizer.zero_grad()
        loss_dm.backward()
        self.dynamic_model_optimizer.step()


    
        
class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_directions=2):
        super(BidirectionalLSTM, self).__init__()
        self.num_directions = num_directions
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=False, batch_first=True)

    def forward(self, x, h):
        # x: (batch_size, seq_len, input_size)
        out, h = self.lstm(x, h)
        # out: (batch_size, seq_len, num_directions * hidden_size)
        return out, h
    

class Decoder_Net(nn.Module):
    def __init__(self, input_size, output_size, batch_size):
        super(Decoder_Net, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = 128
        self.num_layers = 1
        self.lstm = BidirectionalLSTM(input_size, self.hidden_size, num_layers=self.num_layers)
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(self.hidden_size, 64)  
        self.fc2 = nn.Linear(64, output_size)
        self.random_hidden = ((torch.randn(self.num_layers, self.batch_size, self.hidden_size) * 0.01 + 0.0).to(device), (torch.randn(self.num_layers, batch_size, self.hidden_size)*0.001).to(device))
        self.hidden = self.random_hidden

        

    def forward(self, x):
        out, self.hidden = self.lstm(x.unsqueeze(1), self.hidden)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.fc2(out[:, -1, :])
        
        return out

    def reset_hidden(self, batch_size, flag=False):
        if not flag:
            self.hidden = ((torch.randn(self.num_layers, batch_size, self.hidden_size) * 0.01 + 0.0).to(device), (torch.randn(self.num_layers, batch_size, self.hidden_size)*0.001).to(device))
        else:
            self.hidden = ((torch.randn(self.num_layers, batch_size, self.hidden_size) * 0.01 + 0.0).to(device), (torch.randn(self.num_layers, batch_size, self.hidden_size)*0.001).to(device))
            return self.hidden
    


class Encoder_Net(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 128)
        self.linear4 = nn.Linear(128, 256)
        self.linear5 = nn.Linear(256, output_size)
        
       
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)

        return x


class MPC_Policy_Net(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 64)
        # self.bn1 = nn.BatchNorm1d(64)
        self.linear2 = nn.Linear(64, 128)
        # self.bn2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, 64)
        # self.bn3 = nn.BatchNorm1d(64)
        self.linear4 = nn.Linear(64, 32)
        # self.bn4 = nn.BatchNorm1d(32)
        self.linear5 = nn.Linear(32, output_size)
        
       
    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = F.gelu(self.linear2(x))
        x = F.gelu(self.linear3(x))
        x = F.gelu(self.linear4(x))
        x = F.tanh(self.linear5(x))

        return x

class Dynamic_Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Dynamic_Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 250)
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, 250)  
        self.fc4 = nn.Linear(250, 250)
        self.fc5 = nn.Linear(250, 250)
        self.fc6 = nn.Linear(250, output_size)

        

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))
        out = self.fc6(out)
        
        return out


    
class MPC(object):
    def __init__(self, h=20, N=50, argmin=True, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.d = self.target_action_space_dim      # dimension of function input X
        self.h = h                        # sequence length of sample data
        self.N = N                        # sample N examples each iteration
        self.reverse = argmin         # try to maximum or minimum the target function
        self.v_min = [-1.0] * self.d                # the value minimum
        self.v_max = [1.0] * self.d                 # the value maximum

        # state AE
        self.decoder_net = Decoder_Net(self.target_state_space_dim, self.source_state_space_dim, self.batch_size).to(device)
        self.encoder_net = Encoder_Net(self.source_state_space_dim, self.target_state_space_dim).to(device)
        self.dec_optimizer = optim.Adam(self.decoder_net.parameters(), lr=self.lr)
        self.enc_optimizer = optim.Adam(self.encoder_net.parameters(), lr=self.lr)
        
        # mpc policy net
        self.mpc_policy_net = self.agent_target.mpc_policy_net
        self.dynamic_model = self.agent_target.dynamic_model
    
    def learn(self, train_set):
        loss_function = nn.MSELoss()
        index = list(range(train_set.buffer_len()))
        combinations_list = list(combinations(index, 2))
        selected_combinations = random.sample(combinations_list, self.batch_size)
        state_a, next_state_a, action_a, state_b, next_state_b, action_b = [], [], [], [], [], []
        for combo in selected_combinations:
            traj_a, traj_b = train_set.sample(combo)
            traj_a, traj_b = compare_trajectories(traj_a, traj_b)
            # if random.random() < 0.1:
            #     traj_a, traj_b = traj_b, traj_a
            state_a_tensor = torch.tensor(traj_a['state'], dtype=torch.float32)
            next_state_a_tensor = torch.tensor(traj_a['next_state'], dtype=torch.float32)
            action_a_tensor = torch.tensor(traj_a['action'], dtype=torch.float32)
            
            state_b_tensor = torch.tensor(traj_b['state'], dtype=torch.float32)
            next_state_b_tensor = torch.tensor(traj_b['next_state'], dtype=torch.float32)
            action_b_tensor = torch.tensor(traj_b['action'], dtype=torch.float32)

            state_a.append(state_a_tensor)
            next_state_a.append(next_state_a_tensor)
            action_a.append(action_a_tensor)
            state_b.append(state_b_tensor)
            next_state_b.append(next_state_b_tensor)
            action_b.append(action_b_tensor)

        state_a = torch.stack(state_a, dim=0).to(device)
        next_state_a = torch.stack(next_state_a, dim=0).to(device)
        action_a = torch.stack(action_a, dim=0).to(device)
        state_b = torch.stack(state_b, dim=0).to(device)
        next_state_b = torch.stack(next_state_b, dim=0).to(device)
        action_b = torch.stack(action_b, dim=0).to(device)
        

        # update state decoder
        ### traj a
        env = gym.make(args.source_env)
        env.reset(seed = args.seed)
        loss_tran_a = 0
        loss_rec_a = 0
        R_s_a_tensor = torch.zeros((self.batch_size, 1)).to(device)
        self.decoder_net.reset_hidden(self.batch_size, flag=True)
        dec_s = self.decoder_net(state_a[:,0,:].squeeze(1))
        for i in range(env.max_episode_steps):
            tran_s1 = []
            for b in range(self.batch_size):
                env.reset_specific(state = dec_s[b].cpu().detach().numpy())
                action = self.agent.get_action_val(dec_s[b].cpu())
                tmp = env.step(np.squeeze(action))
                tran_s1_tensor = torch.tensor(tmp[0], dtype=torch.float32).unsqueeze(0).to(device)
                tran_s1.append(tran_s1_tensor)
                ##
                r = compute_source_R(dec_s[b], action)
                R_s_a_tensor[b] = R_s_a_tensor[b] + (0.99**i)*r
            tran_s1 = torch.stack(tran_s1, dim=0).squeeze(1)
            dec_s1 = self.decoder_net(next_state_a[:,i,:].squeeze(1))
            loss_tran_a += loss_function(tran_s1, dec_s1)
            rec_s = self.encoder_net(dec_s)  ###
            loss_rec_a += loss_function(rec_s, state_a[:,i,:].squeeze(1))  ###
            dec_s = dec_s1
        env.close()
        
        ### traj b
        env = gym.make(args.source_env)
        env.reset(seed = args.seed)
        loss_tran_b = 0
        loss_rec_b = 0
        R_s_b_tensor = torch.zeros((self.batch_size, 1)).to(device)
        self.decoder_net.reset_hidden(self.batch_size, flag=True)
        dec_s = self.decoder_net(state_b[:,0,:].squeeze(1))
        for i in range(env.max_episode_steps):
            tran_s1 = []
            for b in range(self.batch_size):
                env.reset_specific(state = dec_s[b].cpu().detach().numpy())
                action = self.agent.get_action_val(dec_s[b].cpu())
                tmp = env.step(np.squeeze(action))
                tran_s1_tensor = torch.tensor(tmp[0], dtype=torch.float32).unsqueeze(0).to(device)
                tran_s1.append(tran_s1_tensor)
                ##
                r = compute_source_R(dec_s[b], action)
                R_s_b_tensor[b] = R_s_b_tensor[b] + (0.99**i)*r
                
            tran_s1 = torch.stack(tran_s1, dim=0).squeeze(1)
            dec_s1 = self.decoder_net(next_state_b[:,i,:].squeeze(1))
            loss_tran_b += loss_function(tran_s1, dec_s1)
            rec_s = self.encoder_net(dec_s)  ###
            loss_rec_b += loss_function(rec_s, state_b[:,i,:].squeeze(1))  ###
            dec_s = dec_s1
        env.close()

        ## transition loss
        loss_tran = (loss_tran_a+loss_tran_b) / env.max_episode_steps

        ## rec loss
        loss_rec = (loss_rec_a+loss_rec_b) / env.max_episode_steps
        
        ## preference consistency loss
        result_tensor = torch.cat((R_s_a_tensor, R_s_b_tensor), dim=-1).type(torch.float32)#.unsqueeze(0)
        sub_first_rewards = result_tensor-result_tensor[:,0][:,None]
        loss_pref = torch.sum(sub_first_rewards.exp(), -1).log().mean()
        
        dec_loss = loss_tran + loss_rec + loss_pref
        enc_loss = loss_rec

        self.dec_optimizer.zero_grad()
        self.enc_optimizer.zero_grad()
        dec_loss.backward(retain_graph=True)
        enc_loss.backward(retain_graph=True)
        self.dec_optimizer.step()
        self.enc_optimizer.step()

        
        return loss_tran.item(), loss_pref.item(), loss_rec.item()
    

    def __syntheticTraj(self, state):
        self.mpc_policy_net.eval()
        action = self.mpc_policy_net(torch.tensor(state, dtype=torch.float32).to(device))

        traj_state = []
        traj_action = torch.zeros((self.N, self.h, self.target_action_space_dim)).to(device)
        for i in range(self.N):
            traj_state_one = []
            traj_state_one.append(state)
            s = state
            if i != 0:
                noise = torch.randn_like(action) * 0.1
                a = action + noise 
                traj_action[i,0,:] = a
            else:
                a = action
                traj_action[i,0,:] = a
            for j in range(1, self.h):
                s = self.dynamic_model(torch.cat([torch.tensor(s, dtype=torch.float32).to(device), a], dim=0))
                traj_state_one.append(s)
                a = self.mpc_policy_net(torch.tensor(s, dtype=torch.float32).to(device))
                traj_action[i,j,:] = a
            s = self.dynamic_model(torch.cat([torch.tensor(s, dtype=torch.float32).to(device), a], dim=0))
            traj_state_one.append(s)

            traj_state.append(traj_state_one)
        traj_state = torch.tensor(traj_state, dtype=torch.float32).to(device)
        return traj_state, traj_action

    
    def __decodeTraj(self, s, a):
        with torch.no_grad():
            env = gym.make(args.source_env)    
            env.reset(seed = args.seed)
            traj_action = []
            reward_list = []
            for i in range(self.N):
                self.decoder_net.reset_hidden(1)
                traj_action_one = []
                reward = 0
                state = self.decoder_net(s[i,0,:].unsqueeze(0)).squeeze(0)
                env.reset_specific(state = state.cpu().detach().numpy())
                for j in range(self.h):
                    action = self.agent.get_action_val(state.cpu())
                    r = compute_source_R(state, action).cpu().detach().numpy()
                    reward += r
                    next_state = self.decoder_net(s[i,j+1,:].unsqueeze(0)).squeeze(0)
                    traj_action_one.append(a[i,j,:])
                    state = next_state
                traj_action.append(traj_action_one)
                reward_list.append(reward)

        data = [(xi, yi) for xi, yi in zip(traj_action, reward_list)]
        sorted_action = sorted(data, key=lambda pair: pair[1], reverse=self.reverse)

        rank_indices = np.argsort(reward_list)[::-1]
        rank_list = [rank_indices.tolist().index(i) + 1 for i in range(len(rank_indices))]
    
        # find corresponding index in unsorted traj_action
        ind = 0
        for index, element in enumerate(traj_action):
            if all(torch.all(torch.eq(elem, targ)) for elem, targ in zip(element, sorted_action[0][0])):
                ind = index
        
        sorted_action = [torch.stack(item[0]) for item in sorted_action]
        sorted_action = torch.stack(sorted_action)
        self.rank_list_source_all.append(rank_list)
        

        return sorted_action, rank_list, ind

        
    
    def eval(self, state):
        self.rank_list_source_all = []
        self.rank_list_target_all = []
        self.rank_top_all = []

        # generate action sequence using policy network and get state sequence
        s, a = self.__syntheticTraj(state)
        # decode state sequence to source state sequence and get sorted action sequence (in terms of reward)
        sorted_a, rank_list_source, ind = self.__decodeTraj(s, a)

        # return first action of the best sequence
        return sorted_a[0,0,:], self.rank_list_source_all, self.rank_list_target_all, self.rank_top_all

        
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compare_trajectories(trajectory_info_a, trajectory_info_b):
    if trajectory_info_a['total_rewards'] > trajectory_info_b['total_rewards']:
        return trajectory_info_a, trajectory_info_b
    elif trajectory_info_a['total_rewards'] < trajectory_info_b['total_rewards']:
        return trajectory_info_b, trajectory_info_a
    else:
        if trajectory_info_a['count'] > trajectory_info_b['count']:
            return trajectory_info_a, trajectory_info_b
        elif trajectory_info_a['count'] < trajectory_info_b['count']:
            return trajectory_info_b, trajectory_info_a
        else:
            if trajectory_info_a['dist'] < trajectory_info_b['dist']:
                return trajectory_info_a, trajectory_info_b
            elif trajectory_info_a['dist'] > trajectory_info_b['dist']:
                return trajectory_info_b, trajectory_info_a
            else:
                return trajectory_info_a, trajectory_info_b            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int, nargs='?', default=2)
    parser.add_argument("source_env", type=str, nargs='?', default="Reacher-v4")
    parser.add_argument("target_env", type=str, nargs='?', default="Reacher-3joints")
    parser.add_argument("source_ep", type=int, nargs='?', default=10000)
    parser.add_argument("targetPolicy_ep", type=int, nargs='?', default=10000)
    parser.add_argument("targetData_ep", type=int, nargs='?', default=10000)
    parser.add_argument("MPC_pre_ep", type=int, nargs='?', default=10000)
    parser.add_argument("expert_ratio", type=float, nargs='?', default=0.2)
    parser.add_argument("random_ratio", type=float, nargs='?', default=0.8)
    parser.add_argument("decoder_batch", type=int, nargs='?', default=32)
    parser.add_argument("decoder_ep", type=int, nargs='?', default=500)
    parser.add_argument("device", type=str, nargs='?', default="cuda:1")
    # parser.add_argument("seed", type=int, nargs='?', default=2)
    # parser.add_argument("source_env", type=str, nargs='?', default="Reacher-v4")
    # parser.add_argument("target_env", type=str, nargs='?', default="Reacher-3joints")
    # parser.add_argument("source_ep", type=int, nargs='?', default=10000)
    # parser.add_argument("targetPolicy_ep", type=int, nargs='?', default=10000)
    # parser.add_argument("targetData_ep", type=int, nargs='?', default=10)
    # parser.add_argument("MPC_pre_ep", type=int, nargs='?', default=10000)
    # parser.add_argument("expert_ratio", type=float, nargs='?', default=0.8)
    # parser.add_argument("random_ratio", type=float, nargs='?', default=0.2)
    # parser.add_argument("decoder_batch", type=int, nargs='?', default=2)
    # parser.add_argument("decoder_ep", type=int, nargs='?', default=500)
    # parser.add_argument("device", type=str, nargs='?', default="cuda:2")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = args.device

    ##### Loading source domain policy #####
    print("##### Loading source domain policy #####")
    
    env = gym.make(args.source_env)
    env_target = gym.make(args.target_env)

    # Params
    tau = 0.01
    gamma = 0.99
    q_lr = 3e-4
    policy_lr = 3e-4
    alpha_lr = 3e-4
    buffer_maxlen = 1000000
 
    batch_size = 128 
    agent = SAC(env, gamma, tau, buffer_maxlen, q_lr, policy_lr, alpha_lr)
    
    if os.path.exists(str(args.seed)+'_reacherModel.pth'):
        agent.policy_net.load_state_dict(torch.load(str(args.seed)+'_reacherModel.pth'))
    else:
        Return = [] 
        for episode in range(args.source_ep):
            score = 0
            state = env.reset(seed = args.seed)[0]
            for time_steps in range(env.max_episode_steps):
                if random.uniform(0, 1) < 0.1: 
                    action = np.random.uniform(low=-1, high=1, size=(env.action_space.shape[0],))
                else:
                    action = agent.get_action(state)
                next_state, reward, done, _, info = env.step(action)

                if(time_steps==49 or np.any(np.isinf(state))):
                    done = True
                done_mask = 0.0 if done else 1.0
                agent.buffer.push((state, action, reward, next_state, done_mask))
                state = next_state

                score += reward
                if done:
                    break
                if agent.buffer.buffer_len() > batch_size:
                    agent.update(batch_size)        
            print("episode:{}, Return:{}, buffer_capacity:{}".format(episode, score, agent.buffer.buffer_len()))
            Return.append(score)
        torch.save(agent.policy_net.state_dict(), str(args.seed)+'_reacherModel.pth')
        env.close()


    ##### Loading target domain expert policy #####
    print("##### Loading target domain expert policy #####")
    env_target = gym.make(args.target_env)
 
    # Params
    tau = 0.01
    gamma = 0.99
    q_lr = 3e-4
    policy_lr = 3e-4
    alpha_lr = 3e-4
    buffer_maxlen = 1000000
 
    batch_size = 128

    agent_target = SAC_target(env_target, gamma, tau, buffer_maxlen, q_lr, policy_lr, alpha_lr)
    if os.path.exists("reacher_targetModel.pth"):
        agent_target.policy_net.load_state_dict(torch.load(str(args.seed)+'_reacher_targetModel.pth'))
    else:
        Return = []
        for episode in range(args.targetPolicy_ep):
            score = 0
            state = env_target.reset(seed = args.seed)[0]
            for time_steps in range(env_target.max_episode_steps):
                action = agent_target.get_action(state)
                next_state, reward, done, _, info = env_target.step(action)
                if(time_steps==49 or np.any(np.isinf(state))):
                    done = True
                done_mask = 0.0 if done else 1.0
                agent_target.buffer.push((state, action, reward, next_state, done_mask))
                state = next_state

                score += reward
                if done:
                    break
                if agent_target.buffer.buffer_len() > batch_size:
                    agent_target.update(batch_size)

            print("episode:{}, Return:{}, buffer_capacity:{}".format(episode, score, agent_target.buffer.buffer_len()))
            Return.append(score)
            torch.save(agent_target.policy_net.state_dict(), str(args.seed)+'_reacher_targetModel.pth')
        env_target.close()

    ##### Collecting target domain data #####
    print("##### Collecting target domain data #####")

    print("--------- expert data ---------")
    ## expert data
    env_target = gym.make(args.target_env)
    train_set = ReplayBuffer_target()

    expert_ratio = args.expert_ratio
    random_ratio = args.random_ratio
    
    # min_reward = float('inf')
    # min_index = -1
    STATE = []
    NEXT_STATE = []
    ACTION = []
    REWARD = []
    Return = []
    max_episode = args.targetData_ep 
    for episode in range(int(max_episode*expert_ratio)):
        score = 0
        count = 0
        info = 0
        tmp = env_target.reset()
        state = tmp[0]
        state_list = []
        next_state_list = []
        action_list = []
        for time_steps in range(env_target.max_episode_steps):
            action = agent_target.get_action_val(state)
            tmp = env_target.step(action)
            next_state = tmp[0]
            reward = tmp[1]
            done = tmp[2]
            info = tmp[4]
            if np.abs(info['reward_dist']) == 0.0:
                count += 1
            if(time_steps==49 or np.any(np.isinf(state))):
                done = True
            done_mask = 0.0 if done else 1.0
            state_list.append(state)
            next_state_list.append(next_state)
            action_list.append(action)
            agent_target.buffer_.push((state, action, reward, next_state, done_mask))
            state = next_state

            score += reward
            if done:
                break
        
        STATE.append(state_list)
        NEXT_STATE.append(next_state_list)
        ACTION.append(action_list)
        # if len(STATE) >= int(max_episode*0.1) and score > min_reward:
        #     STATE[min_index] = state_list
        #     NEXT_STATE[min_index] = next_state_list
        #     ACTION[min_index] = action_list
        #     REWARD[min_index] = score
        #     min_reward = min(REWARD)
        #     min_index = REWARD.index(min_reward)

        # else:
        #     STATE.append(state_list)
        #     NEXT_STATE.append(next_state_list)
        #     ACTION.append(action_list)
        #     REWARD.append(score)
        #     min_reward = min(REWARD)
        #     min_index = REWARD.index(min_reward)


        
        train_set.push(score, count, np.abs(info['reward_dist']), state_list, next_state_list, action_list)
        print("episode:{}, Return:{}, buffer_capacity:{}".format(episode, score, agent_target.buffer.buffer_len()))
        Return.append(score)
    env_target.close()
    

    print("--------- random data ---------")
    ## random data
    env_target = gym.make(args.target_env)
 
    Return = []
    max_episode = args.targetData_ep 
    
    for episode in range(int(max_episode*random_ratio)):
        score = 0
        count = 0
        info = 0
        tmp = env_target.reset()
        state = tmp[0]
        state_list = []
        next_state_list = []
        action_list = []
        for time_steps in range(env_target.max_episode_steps):
            action = np.random.uniform(low=-1, high=1, size=(env_target.action_space.shape[0],))
            tmp = env_target.step(action)
            next_state = tmp[0]
            reward = tmp[1]
            done = tmp[2]
            info = tmp[4]
            if np.abs(info['reward_dist']) == 0.0:
                count += 1
            if(time_steps==49 or np.any(np.isinf(state))):
                done = True
            done_mask = 0.0 if done else 1.0
            state_list.append(state)
            next_state_list.append(next_state)
            action_list.append(action)
            agent_target.buffer_.push((state, action, reward, next_state, done_mask))
            state = next_state

            score += reward
            if done:
                break
        
        STATE.append(state_list)
        NEXT_STATE.append(next_state_list)
        ACTION.append(action_list)
        # if len(STATE) >= int(max_episode*0.1) and score > min_reward:
        #     STATE[min_index] = state_list
        #     NEXT_STATE[min_index] = next_state_list
        #     ACTION[min_index] = action_list
        #     REWARD[min_index] = score
        #     min_reward = min(REWARD)
        #     min_index = REWARD.index(min_reward)

        train_set.push(score, count, np.abs(info['reward_dist']), state_list, next_state_list, action_list)
        print("episode:{}, Return:{}, buffer_capacity:{}".format(episode, score, agent_target.buffer.buffer_len()))
        Return.append(score)
    env_target.close()

    STATE = np.vstack([np.array(lst) for lst in STATE])
    NEXT_STATE = np.vstack([np.array(lst) for lst in NEXT_STATE])
    ACTION = np.vstack([np.array(lst) for lst in ACTION])
    
    # np.save(str(args.seed)+'_now_state.npy', STATE)
    # np.save(str(args.seed)+'_next_state.npy', NEXT_STATE)
    # np.save(str(args.seed)+'_action.npy', ACTION)
    
    
    print("Amount of expert data : ", int(max_episode*expert_ratio))
    print("Amount of random data : ", int(max_episode*random_ratio))


    ##### Loading MPC policy and Dynamic Model #####
    print("##### Loading MPC policy and Dynamic Model #####")
    # for i in range(args.MPC_pre_ep):
    #     agent_target.update_(batch_size)
    #     torch.save(agent_target.mpc_policy_net.state_dict(), str(args.seed)+'_MPCModel.pth')
    #     torch.save(agent_target.dynamic_model.state_dict(), str(args.seed)+'_DynamicModel.pth')
    agent_target.mpc_policy_net.load_state_dict(torch.load(str(args.seed)+'_MPCModel.pth'))
    agent_target.dynamic_model.load_state_dict(torch.load(str(args.seed)+'_DynamicModel.pth'))


    ##### Training state decoder #####
    print("##### Training state decoder #####")
    env = gym.make(args.source_env)#, render_mode='human')
    env_target = gym.make(args.target_env)#, render_mode='human')

    params = {
        'batch_size': args.decoder_batch,
        'lr': 0.001,  
        'source_state_space_dim': env.observation_space.shape[0],
        'target_state_space_dim': env_target.observation_space.shape[0],
        'source_action_space_dim': env.action_space.shape[0],
        'target_action_space_dim': env_target.action_space.shape[0],
        'agent': agent,
        'agent_target': agent_target,
    }
    mpc = MPC(**params)

    batch_size = args.decoder_batch
    Return_val = []
    
    # val state decoder
    # ep 0
    mpc.decoder_net.eval()
    tmp = env_target.reset(seed = args.seed)
    s0 = tmp[0]
    total_reward = 0
    rank_source_list_all, rank_target_list_all, rank_top_list_all = [], [], []
    rank_source_list, rank_target_list, rank_top_list = [], [], []
    for time_steps in range(env_target.max_episode_steps):
        a0_target, rank_source, rank_target, rank_top = mpc.eval(s0)
        rank_source_list.append(rank_source)
        rank_target_list.append(rank_target)
        rank_top_list.append(rank_top)
        tmp = env_target.step(a0_target.cpu().detach().numpy())
        s1 = tmp[0]
        r1 = tmp[1]
        done = tmp[2]
        total_reward += r1
        if(time_steps==49 or np.any(np.isinf(s0))):
            done = True
    
        if done:
            break
        
        s0 = s1

    # wandb.log({"avg validation_reward": total_reward, 
    #           })
    rank_source_list_all.append(rank_source_list)
    rank_target_list_all.append(rank_target_list)
    rank_top_list_all.append(np.mean(rank_top_list))
    Return_val.append(total_reward)
    print(f'episode: {0}, validation reward: {total_reward}')
    
        
    for j in range(1, args.decoder_ep+1):
        loss_tran_list, loss_pref_list, loss_rec_list = [], [], []
        for k in range(1):
            # train state decoder
            mpc.decoder_net.train()
            loss_tran, loss_pref, loss_rec = mpc.learn(train_set)
            loss_tran_list.append(loss_tran)
            loss_pref_list.append(loss_pref)
            loss_rec_list.append(loss_rec)

        print(f'episode: {j}, transition loss: {np.mean(loss_tran_list)}, pref loss: {np.mean(loss_pref_list)}, rec loss: {np.mean(loss_rec_list)}')

        # val state decoder
        mpc.decoder_net.eval()
        return_list_ep = []
        rd_list_ep = []
        ra_list_ep = []
        for i in range(1):
            tmp = env_target.reset(seed = args.seed)
            s0 = tmp[0]
            total_reward = 0
            rd = 0
            ra = 0
            rank_source_list, rank_target_list, rank_top_list = [], [], []
            for time_steps in range(env_target.max_episode_steps):
                a0_target, rank_source, rank_target, rank_top = mpc.eval(s0)
                rank_source_list.append(rank_source)
                rank_target_list.append(rank_target)
                rank_top_list.append(rank_top)
                tmp = env_target.step(a0_target.cpu().detach().numpy())
                s1 = tmp[0]
                r1 = tmp[1]
                done = tmp[2]
                info = tmp[4]
                rd += info['reward_dist']
                ra += info['reward_ctrl']
                total_reward += r1
                if(time_steps==49 or np.any(np.isinf(s0))):
                    done = True
                

                if done:
                    break
                
                s0 = s1
            rd_list_ep.append(rd)
            ra_list_ep.append(ra)
            return_list_ep.append(total_reward)  
            rank_source_list_all.append(rank_source_list)
            rank_target_list_all.append(rank_target_list)
            rank_top_list_all.append(np.mean(rank_top_list))

        print(f'episode: {j}, avg. validation reward: {np.mean(return_list_ep)}')
        Return_val.append(np.mean(return_list_ep))
        # wandb.log({"avg validation_reward": np.mean(return_list_ep), 
        #           "tran loss": np.mean(loss_tran_list),
        #           "pref loss": np.mean(loss_pref_list),
        #           "rec loss": np.mean(loss_rec_list),
        #           "reward_distance": np.mean(rd_list_ep),
        #           "reward_ctrl": np.mean(ra_list_ep),
        #           })
        filename = './data/'+str(args.source_env)+'/'+str(args.seed)+'_0.2_0.8.npz'
        np.savez(filename, reward_val = Return_val)