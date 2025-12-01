import gym
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config
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
from rlkit.torch.sac.policies import TanhGaussianPolicy
import json
# import wandb
# SAC credit to https://github.com/ARISE-Initiative/robosuite-benchmark/tree/master

warnings.filterwarnings('ignore')

# wandb.init(project="cdpc", name = 'lift')

def load_variant():
        def get_expl_env_kwargs():
            """
            Grabs the robosuite-specific arguments and converts them into an rlkit-compatible dict for exploration env
            """
            env_kwargs = dict(
                env_name=args.env_name_t,
                robots=args.robots_t,
                horizon=500,
                control_freq=20,
                controller="OSC_POSE",
                reward_scale=1.0,
                hard_reset=False,
                ignore_done=True,
            )

            # Add in additional ones that may not always be specified
            # if args.env_config is not None:
            #     env_kwargs["env_configuration"] = args.env_config
            # if args.prehensile is not None:
            #     env_kwargs["prehensile"] = BOOL_MAP[args.prehensile.lower()]

            # Lastly, return the dict
            return env_kwargs

        def get_eval_env_kwargs():
            """
            Grabs the robosuite-specific arguments and converts them into an rlkit-compatible dict for evaluation env
            """
            env_kwargs = dict(
                env_name=args.env_name_t,
                robots=args.robots_t,
                horizon=500,
                control_freq=20,
                controller="OSC_POSE",
                reward_scale=1.0,
                hard_reset=False,
                ignore_done=True,
            )

            # Add in additional ones that may not always be specified
            # if args.env_config is not None:
            #     env_kwargs["env_configuration"] = args.env_config
            # if args.prehensile is not None:
            #     env_kwargs["prehensile"] = BOOL_MAP[args.prehensile.lower()]

            # Lastly, return the dict
            return env_kwargs

        trainer_kwargs = dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3e-4,
            qf_lr=3e-4,
            reward_scale=1.0,
            use_automatic_entropy_tuning=(not False),
        )
        variant = dict(
            algorithm="SAC",
            seed=2,
            version="normal",
            replay_buffer_size=int(1E6),
            qf_kwargs=dict(
                hidden_sizes=[256, 256],
            ),
            policy_kwargs=dict(
                hidden_sizes=[256, 256],
            ),
            algorithm_kwargs=dict(
                num_epochs=2000,
                num_eval_steps_per_epoch=500 * 10,
                num_trains_per_train_loop=1000,
                num_expl_steps_per_train_loop=500 * 10,
                min_num_steps_before_training=1000,
                expl_max_path_length=500,
                eval_max_path_length=500,
                batch_size=256,
            ),
            trainer_kwargs=trainer_kwargs,
            expl_environment_kwargs=get_expl_env_kwargs(),
            eval_environment_kwargs=get_eval_env_kwargs(),
        )

        return variant

def make_env(env_name, robots):
    controller_config = load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        env_name=env_name, # try with other tasks like "Stack" and "Door"
        robots=robots,  # try with other robots like "Sawyer" and "Jaco"
        gripper_types="default",                # use default grippers per robot arm
        controller_configs=controller_config,   # each arm is controlled using OSC
        has_renderer=False,                     # no on-screen rendering
        has_offscreen_renderer=False,           # no off-screen rendering
        control_freq=20,                        # 20 hz control for applied actions
        horizon=500,                            # each episode terminates after 200 steps
        use_object_obs=True,                    # provide object observations to agent
        use_camera_obs=False,                   # don't provide image observations to agent
        reward_shaping=True,                    # use a dense reward signal for learning
        hard_reset=False,
        # ignore_done = True,
        reward_scale=1.0, 
    )
    env = GymWrapper(env)
    return env

def compute_source_R(next_state):
    reward = 0.0
    if next_state[2] > round(next_state[2], 1) + 0.04:
        reward = 2.25
    else:
        dist = np.linalg.norm(next_state[0:3] - next_state[31:34])
        reaching_reward = 1 - np.tanh(10.0 * dist)
        reward += reaching_reward
    reward *= 1.0/2.25
    return reward

def compute_source_R_(next_state):
    reward = 0.0
    dist = torch.norm(next_state[0:3] - next_state[31:34])
    reaching_reward = 1 - torch.tanh(10.0 * dist)
    reward += reaching_reward#.item()
    return reward

def compute_target_R(next_state):
    reward = 0.0
    if next_state[2] > round(next_state[2], 1) + 0.04:
        reward = 2.25
    else:
        dist = np.linalg.norm(next_state[0:3] - next_state[31:34])
        reaching_reward = 1 - np.tanh(10.0 * dist)
        reward += reaching_reward
    reward *= 1.0/2.25
    return reward


def compare_trajectories(trajectory_info_a, trajectory_info_b):
    if trajectory_info_a['total_rewards'] >= trajectory_info_b['total_rewards']:
        return trajectory_info_a, trajectory_info_b
    elif trajectory_info_a['total_rewards'] < trajectory_info_b['total_rewards']:
        return trajectory_info_b, trajectory_info_a
    else:
        return trajectory_info_a, trajectory_info_b            

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

    def push(self, total_rewards, state, next_state, action):
        trajectory_info = {
        'total_rewards': total_rewards,
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
        self.hidden = ((torch.randn(self.num_layers, self.batch_size, self.hidden_size) * 0.01 + 0.0).to(device), (torch.randn(self.num_layers, batch_size, self.hidden_size)*0.001).to(device))
        

    def forward(self, x):
        out, self.hidden = self.lstm(x, self.hidden)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out, self.hidden

    def reset_hidden(self, batch_size):
        self.hidden = ((torch.randn(self.num_layers, batch_size, self.hidden_size) * 0.01 + 0.0).to(device), (torch.randn(self.num_layers, batch_size, self.hidden_size)*0.001).to(device))
        
class Encoder_Net(nn.Module):
    def __init__(self, input_size, output_size, batch_size):
        super(Encoder_Net, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = 128
        self.num_layers = 1
        self.lstm = BidirectionalLSTM(input_size, self.hidden_size, num_layers=self.num_layers)
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(self.hidden_size, 64)  
        self.fc2 = nn.Linear(64, output_size)
        self.hidden = ((torch.randn(self.num_layers, self.batch_size, self.hidden_size) * 0.01 + 0.0).to(device), (torch.randn(self.num_layers, batch_size, self.hidden_size)*0.001).to(device))
        

    def forward(self, x):
        out, self.hidden = self.lstm(x, self.hidden)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out, self.hidden

    def reset_hidden(self, batch_size):
        self.hidden = ((torch.randn(self.num_layers, batch_size, self.hidden_size) * 0.01 + 0.0).to(device), (torch.randn(self.num_layers, batch_size, self.hidden_size)*0.001).to(device))


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
    def __init__(self, h=10, N=100, argmin=True, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.d = self.target_action_space_dim      # dimension of function input X
        self.h = h                        # sequence length of sample data
        self.N = N                        # sample N examples each iteration
        self.reverse = argmin         # try to maximum or minimum the target function
        self.v_min = [-1.0] * self.d      # the value minimum
        self.v_max = [1.0] * self.d      # the value maximum

        # state AE
        self.decoder_net = Decoder_Net(self.target_state_space_dim, self.source_state_space_dim, self.batch_size).to(device)
        self.encoder_net = Encoder_Net(self.source_state_space_dim, self.target_state_space_dim, self.batch_size).to(device)
        self.dec_optimizer = optim.Adam(self.decoder_net.parameters(), lr=self.lr, weight_decay=1e-4)
        self.enc_optimizer = optim.Adam(self.encoder_net.parameters(), lr=self.lr, weight_decay=1e-4)
        

    def learn(self, train_set):
        loss_function = nn.MSELoss()
        index = list(range(train_set.buffer_len()))
        combinations_list = list(combinations(index, 2))
        selected_combinations = random.sample(combinations_list, self.batch_size)
        state_a, next_state_a, state_b, next_state_b = [], [], [], []
        for combo in selected_combinations:
            traj_a, traj_b = train_set.sample(combo)
            traj_a, traj_b = compare_trajectories(traj_a, traj_b)
            state_a_tensor = torch.tensor(traj_a['state'], dtype=torch.float32)
            next_state_a_tensor = torch.tensor(traj_a['next_state'], dtype=torch.float32)
            
            state_b_tensor = torch.tensor(traj_b['state'], dtype=torch.float32)
            next_state_b_tensor = torch.tensor(traj_b['next_state'], dtype=torch.float32)

            state_a.append(state_a_tensor)
            next_state_a.append(next_state_a_tensor)
            state_b.append(state_b_tensor)
            next_state_b.append(next_state_b_tensor)
        
        state_a = torch.stack(state_a, dim=0).to(device)
        next_state_a = torch.stack(next_state_a, dim=0).to(device)
        state_b = torch.stack(state_b, dim=0).to(device)
        next_state_b = torch.stack(next_state_b, dim=0).to(device)

        state_a = torch.cat([state_a, next_state_a[:, -1:, :]], dim=1)
        state_b = torch.cat([state_b, next_state_b[:, -1:, :]], dim=1)


        # update state decoder
        ### traj a
        R_s_a_tensor = torch.zeros((self.batch_size, 1)).to(device)
        tran_s1_a = torch.empty(0).to(device)
        self.decoder_net.reset_hidden(self.batch_size)
        self.encoder_net.reset_hidden(self.batch_size)
        dec_s_a, _ = self.decoder_net(state_a) # batch_size, seq_len, source_dim
        rec_s_a, _ = self.encoder_net(dec_s_a)
        for b in range(self.batch_size):
            s1 = []
            for i in range(500):
                action_tensor = self.source_policy(dec_s_a[b,i,:].cpu())[0]
                s1_tensor = self.dynamic_model(torch.cat([dec_s_a[b,i,:], action_tensor.to('cuda:1')], dim=0))
                s1.append(s1_tensor)
                r = compute_source_R_(s1_tensor)
                R_s_a_tensor[b] = R_s_a_tensor[b] + r
            s1 = torch.stack(s1, dim=0).squeeze(1).unsqueeze(0)
            tran_s1_a = torch.cat((tran_s1_a, s1), dim=0)
            env.close()
        
        ### traj b
        R_s_b_tensor = torch.zeros((self.batch_size, 1)).to(device)
        tran_s1_b = torch.empty(0).to(device)
        self.decoder_net.reset_hidden(self.batch_size)
        self.encoder_net.reset_hidden(self.batch_size)
        dec_s_b, _ = self.decoder_net(state_b) # batch_size, seq_len, source_dim
        rec_s_b, _ = self.encoder_net(dec_s_b)
        for b in range(self.batch_size):
            s1 = []
            for i in range(500):
                action_tensor = self.source_policy(dec_s_b[b,i,:].cpu())[0]
                s1_tensor = self.dynamic_model(torch.cat([dec_s_b[b,i,:], action_tensor.to('cuda:1')], dim=0))
                s1.append(s1_tensor)
                r = compute_source_R_(s1_tensor)
                R_s_b_tensor[b] = R_s_b_tensor[b] + r
            s1 = torch.stack(s1, dim=0).squeeze(1).unsqueeze(0)
            tran_s1_b = torch.cat((tran_s1_b, s1), dim=0)
            env.close()

        ## transition loss
        loss_tran = loss_function(tran_s1_a, dec_s_a[:,1:,:]) + loss_function(tran_s1_b, dec_s_b[:,1:,:])

        ## rec loss
        loss_rec = loss_function(rec_s_a, state_a) + loss_function(rec_s_b, state_b)
        
        ## preference consistency loss
        result_tensor = torch.cat((R_s_a_tensor, R_s_b_tensor), dim=-1).type(torch.float32)#.unsqueeze(0)
        sub_first_rewards = result_tensor-result_tensor[:,0][:,None]
        loss_pref = torch.sum(sub_first_rewards.exp(), -1).log().mean()
        
        dec_loss = loss_tran + loss_pref + loss_rec
        enc_loss = loss_rec

        self.dec_optimizer.zero_grad()
        self.enc_optimizer.zero_grad()
        dec_loss.backward(retain_graph=True)
        enc_loss.backward(retain_graph=True)
        self.dec_optimizer.step()
        self.enc_optimizer.step()

    
        return loss_tran.item(), loss_pref.item(), loss_rec.item()
    

    def __syntheticTraj(self, state):
        self.mpc_policy.eval()
        action = self.mpc_policy(torch.tensor(state, dtype=torch.float32).to(device))

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
                s = self.target_dynamic_model(torch.cat([torch.tensor(s, dtype=torch.float32).to(device), a], dim=0))
                traj_state_one.append(s)
                a = self.mpc_policy(torch.tensor(s, dtype=torch.float32).to(device))
                traj_action[i,j,:] = a
            s = self.target_dynamic_model(torch.cat([torch.tensor(s, dtype=torch.float32).to(device), a], dim=0))
            traj_state_one.append(s)

            traj_state.append(traj_state_one)
        traj_state = torch.tensor(traj_state, dtype=torch.float32).to(device)
        return traj_state, traj_action

    def __decodeTraj(self, s, a):
        with torch.no_grad():
            self.decoder_net.reset_hidden(self.N)
            dec_s, _ = self.decoder_net(s)
            traj_action = []
            reward_list = []
            for b in range(self.N):
                reward = 0
                traj_action_one = []
                env = make_env(args.env_name_s, args.robots_s)   
                for i in range(self.h):
                    action = self.source_policy(dec_s[b,i,:].cpu())[0]
                    s1 = self.dynamic_model(torch.cat([dec_s[b,i,:].to(device), action.to(device)], dim=0))
                    r = compute_source_R(s1.cpu().detach().numpy())
                    reward += r
                    traj_action_one.append(a[b,i,:])
                traj_action.append(traj_action_one)
                reward_list.append(reward)
                env.close()
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


class MPC_pre(object):
    def __init__(self, buffermaxlen):
        self.buffer = ReplayBuffer(buffermaxlen)

        # MPC policy net
        self.mpc_policy = MPC_Policy_Net(env_target.observation_space.shape[0], env_target.action_space.shape[0]).to(device)
        self.mpc_policy_optimizer = optim.Adam(self.mpc_policy.parameters(), lr=1e-3)
        
        # target dynamic model
        self.target_dynamic_model = Dynamic_Model(env_target.observation_space.shape[0]+env_target.action_space.shape[0], env_target.observation_space.shape[0]).to(device)
        self.target_dynamic_model_optimizer = optim.Adam(self.target_dynamic_model.parameters(), lr=1e-3, weight_decay=1e-6)

    def update_(self, batch_size):
        # update MPC policy net
        state, action, reward, next_state, done = self.buffer.sample(batch_size)
        pred_action = self.mpc_policy(state)
        loss_mpc = F.mse_loss(pred_action, action)
        self.mpc_policy_optimizer.zero_grad()
        loss_mpc.backward()
        self.mpc_policy_optimizer.step()

        # update target dynamic model
        state, action, reward, next_state, done = self.buffer.sample(batch_size)
        pred_next_state = self.target_dynamic_model(torch.cat([state, action], dim=1))
        # pred_next_state = self.dynamic_model(state, action)
        loss_dm = F.mse_loss(pred_next_state, next_state)
        self.target_dynamic_model_optimizer.zero_grad()
        loss_dm.backward()
        self.target_dynamic_model_optimizer.step()

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int, nargs='?', default=2) 
    parser.add_argument("env_name_s", type=str, nargs='?', default="Lift")
    parser.add_argument("robots_s", type=str, nargs='?', default="Panda")
    parser.add_argument("env_name_t", type=str, nargs='?', default="Lift")
    parser.add_argument("robots_t", type=str, nargs='?', default="IIWA")
    parser.add_argument("targetData_ep", type=int, nargs='?', default=1000)
    parser.add_argument("MPC_DM_ep", type=int, nargs='?', default=10000)
    parser.add_argument("expert_ratio", type=float, nargs='?', default=0.2)
    parser.add_argument("random_ratio", type=float, nargs='?', default=0.8)
    parser.add_argument("decoder_batch", type=int, nargs='?', default=32)
    parser.add_argument("decoder_ep", type=int, nargs='?', default=200)
    parser.add_argument("device", type=str, nargs='?', default="cuda:2")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = args.device

    env = make_env(args.env_name_s, args.robots_s)
    env_target = make_env(args.env_name_t, args.robots_t)
    env_help = make_env(args.env_name_t, args.robots_t)     
    # load source policy
    variant = '/your_path/'+str(args.seed)+'_variant.json'
    with open(variant) as f:
        variant = json.load(f)
    source_policy = TanhGaussianPolicy(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            **variant['policy_kwargs'],
    )
    source_policy.load_state_dict(torch.load('/your_path/'+str(args.seed)+'_Panda'+str(args.env_name_s)+'Model.pth'))

    # load target expert policy
    variant = load_variant()
    target_policy = TanhGaussianPolicy(
            obs_dim=env_target.observation_space.shape[0],
            action_dim=env_target.action_space.shape[0],
            **variant['policy_kwargs'],
    )
    target_policy.load_state_dict(torch.load('/your_path/'+str(args.seed)+'_IIWA'+str(args.env_name_s)+'Model.pth'))

    # load source dynamic model
    dynamic_model = Dynamic_Model(env.observation_space.shape[0]+env.action_space.shape[0], env.observation_space.shape[0]).to(device)
    dynamic_model.load_state_dict(torch.load('/your_path/'+str(args.seed)+'_DynamicModel_source.pth'))

    # collect expert data
    env_target = make_env(args.env_name_t, args.robots_t)
    mpc_pre = MPC_pre(1000000)
    train_set = ReplayBuffer_target()

    expert_ratio = args.expert_ratio
    random_ratio = args.random_ratio
    STATE = []
    NEXT_STATE = []
    ACTION = []
    REWARD = []
 
    Return = []
    max_episode = args.targetData_ep 
    for episode in range(int(max_episode*expert_ratio)):
        score = 0
        state = env_target.reset()[0]
        state_list = []
        next_state_list = []
        action_list = []
        done = False
        while not done:
            action = target_policy.get_action(state)[0]
            next_state, reward, done, y, _ = env_target.step(action)
            
            done_mask = 0.0 if done else 1.0
            mpc_pre.buffer.push((state, action, reward, next_state, done_mask))
            state_list.append(state)
            next_state_list.append(next_state)
            action_list.append(action)
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
        
        train_set.push(score, state_list, next_state_list, action_list)
        print("episode:{}, Return:{}".format(episode, score))
        Return.append(score)
    env_target.close()

    # collect random data
    env_target = make_env(args.env_name_t, args.robots_t)
 
    Return = []
    max_episode = args.targetData_ep 
    
    for episode in range(int(max_episode*random_ratio)):
        score = 0
        state = env_target.reset()[0]
        state_list = []
        next_state_list = []
        action_list = []
        done = False
        while not done:
            action = np.random.uniform(low=-1, high=1, size=(env_target.action_space.shape[0],))
            
            next_state, reward, done, y, _ = env_target.step(action)
            
            done_mask = 0.0 if done else 1.0
            mpc_pre.buffer.push((state, action, reward, next_state, done_mask))
            state_list.append(state)
            next_state_list.append(next_state)
            action_list.append(action)
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
            
        train_set.push(score, state_list, next_state_list, action_list)
        print("episode:{}, Return:{}".format(episode, score))
        Return.append(score)
    env_target.close()

    STATE = np.vstack([np.array(lst) for lst in STATE])
    NEXT_STATE = np.vstack([np.array(lst) for lst in NEXT_STATE])
    ACTION = np.vstack([np.array(lst) for lst in ACTION])
    
    np.save(str(args.seed)+'_'+str(args.env_name_s)+'_now_state.npy', STATE)
    np.save(str(args.seed)+'_'+str(args.env_name_s)+'_next_state.npy', NEXT_STATE)
    np.save(str(args.seed)+'_'+str(args.env_name_s)+'_action.npy', ACTION)
    
    print("Amount of expert data : ", int(max_episode*expert_ratio))
    print("Amount of random data : ", int(max_episode*random_ratio))
    

    # Loading MPC policy and Dynamic Model 
    # batch_size = 256
    # for i in range(args.MPC_DM_ep):
    #     mpc_pre.update_(batch_size)
    #     torch.save(mpc_pre.mpc_policy.state_dict(), str(args.seed)+'_MPCModel.pth')
    #     torch.save(mpc_pre.target_dynamic_model.state_dict(), str(args.seed)+'_DynamicModel_target.pth')
    mpc_pre.mpc_policy.load_state_dict(torch.load('/your_path/'+str(args.seed)+'_MPCModel.pth'))
    mpc_pre.target_dynamic_model.load_state_dict(torch.load('/your_path/'+str(args.seed)+'_DynamicModel_target.pth'))

    # MPC
    env = make_env(args.env_name_s, args.robots_s)
    env_target = make_env(args.env_name_t, args.robots_t)
        
    params = {
        'batch_size': args.decoder_batch,
        'lr': 0.0005,  
        'source_state_space_dim': env.observation_space.shape[0],
        'target_state_space_dim': env_target.observation_space.shape[0],
        'source_action_space_dim': env.action_space.shape[0],
        'target_action_space_dim': env_target.action_space.shape[0],
        'source_policy': source_policy,
        'target_policy': target_policy,
        'mpc_policy': mpc_pre.mpc_policy, 
        'dynamic_model': dynamic_model,
        'target_dynamic_model': mpc_pre.target_dynamic_model,
    }
    mpc = MPC(**params)

    batch_size = args.decoder_batch
    
    # val state decoder
    # ep 0 mpc
    mpc.decoder_net.eval()
    s0 = env_target.reset(seed = args.seed)[0]
    total_reward = 0
    rank_source_list_all, rank_target_list_all, rank_top_list_all = [], [], []
    rank_source_list, rank_target_list, rank_top_list = [], [], []
    done = False
    while not done:
        a0_target, rank_source, rank_target, rank_top = mpc.eval(s0)
        rank_source_list.append(rank_source)
        rank_target_list.append(rank_target)
        rank_top_list.append(rank_top)
        s1, r1, done, y, _ = env_target.step(a0_target.cpu().detach().numpy())
        total_reward += r1

        if done:
            break
        
        s0 = s1

    # wandb.log({"avg validation_reward": total_reward, 
    #           })
    rank_source_list_all.append(rank_source_list)
    rank_target_list_all.append(rank_target_list)
    rank_top_list_all.append(np.mean(rank_top_list))
    print(f'episode: {0}, validation reward: {total_reward}')
    
    # train decoder
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
        for i in range(1):
            s0 = env_target.reset(seed = args.seed)[0]
            total_reward = 0
            rank_source_list, rank_target_list, rank_top_list = [], [], []
            done = False
            while not done:
                a0_target, rank_source, rank_target, rank_top = mpc.eval(s0)
                rank_source_list.append(rank_source)
                rank_target_list.append(rank_target)
                rank_top_list.append(rank_top)
                s1, r1, done, y, _ = env_target.step(a0_target.cpu().detach().numpy())
                
                total_reward += r1
            
                if done:
                    break
                
                s0 = s1
            return_list_ep.append(total_reward)  
            rank_source_list_all.append(rank_source_list)
            rank_target_list_all.append(rank_target_list)
            rank_top_list_all.append(np.mean(rank_top_list))

            
        print(f'episode: {j}, avg. validation reward: {np.mean(return_list_ep)}')
        # wandb.log({"avg validation_reward": np.mean(return_list_ep), 
        #           "tran loss": np.mean(loss_tran_list),
        #           "pref loss": np.mean(loss_pref_list),
        #           "rec loss": np.mean(loss_rec_list),
        #           })
        filename = './data/'+str(args.env_name_s)+'/'+str(args.seed)+'_0.2_0.8.npz'
        np.savez(filename, reward_val = return_list_ep)
        

