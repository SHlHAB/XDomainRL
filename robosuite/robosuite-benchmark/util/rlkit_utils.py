from util.rlkit_custom import rollout
import torch.nn as nn
import torch.nn.functional as F
from rlkit.torch.pytorch_util import set_gpu_mode

import torch
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.td3.td3 import TD3Trainer
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from util.rlkit_custom import CustomTorchBatchRLAlgorithm

from rlkit.core import logger
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper

from robosuite.controllers import load_controller_config, ALL_CONTROLLERS


import numpy as np


# Define agents available
AGENTS = {"SAC", "TD3"}

controller_config = load_controller_config(default_controller="OSC_POSE")
env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="IIWA",  # try with other robots like "Sawyer" and "Jaco"
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



def experiment(variant, agent="SAC"):
    
    # Make sure agent is a valid choice
    assert agent in AGENTS, "Invalid agent selected. Selected: {}. Valid options: {}".format(agent, AGENTS)

    # Get environment configs for expl and eval envs and create the appropriate envs
    # suites[0] is expl and suites[1] is eval
    suites = []
    for env_config in (variant["expl_environment_kwargs"], variant["eval_environment_kwargs"]):
        # Load controller
        controller = env_config.pop("controller")
        if controller in set(ALL_CONTROLLERS):
            # This is a default controller
            controller_config = load_controller_config(default_controller=controller)
        else:
            # This is a string to the custom controller
            controller_config = load_controller_config(custom_fpath=controller)
        # Create robosuite env and append to our list
        suites.append(suite.make(**env_config,
                                 has_renderer=False,
                                 has_offscreen_renderer=False,
                                 use_object_obs=True,
                                 use_camera_obs=False,
                                 reward_shaping=True,
                                 controller_configs=controller_config,
                                 ))
    # Create gym-compatible envs
    expl_env = NormalizedBoxEnv(GymWrapper(suites[0]))
    eval_env = NormalizedBoxEnv(GymWrapper(suites[1]))

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )

    # Define references to variables that are agent-specific
    trainer = None
    eval_policy = None
    expl_policy = None

    # Instantiate trainer with appropriate agent
    if agent == "SAC":
        expl_policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **variant['policy_kwargs'],
        )
        eval_policy = MakeDeterministic(expl_policy)
        mpc_policy = MPC_Policy_Net(env.observation_space.shape[0], env.action_space.shape[0]).to('cuda:0')
        dynamic_model = Dynamic_Model(env.observation_space.shape[0]+env.action_space.shape[0], env.observation_space.shape[0]).to("cuda:0")
        trainer = SACTrainer(
            env=eval_env,
            policy=expl_policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            mpc_policy=mpc_policy,
            dynamic_model=dynamic_model,
            **variant['trainer_kwargs']
        )
        
    elif agent == "TD3":
        eval_policy = TanhMlpPolicy(
            input_size=obs_dim,
            output_size=action_dim,
            **variant['policy_kwargs']
        )
        target_policy = TanhMlpPolicy(
            input_size=obs_dim,
            output_size=action_dim,
            **variant['policy_kwargs']
        )
        es = GaussianStrategy(
            action_space=expl_env.action_space,
            max_sigma=0.1,
            min_sigma=0.1,  # Constant sigma
        )
        expl_policy = PolicyWrappedWithExplorationStrategy(
            exploration_strategy=es,
            policy=eval_policy,
        )
        trainer = TD3Trainer(
            policy=eval_policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            target_policy=target_policy,
            **variant['trainer_kwargs']
        )
    else:
        print("Error: No valid agent chosen!")

    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )

    # Define algorithm
    algorithm = CustomTorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()
    
    


def evaluate_policy(env_config, model_path, n_eval, printout=False):
    if printout:
        print("Loading policy...")

    # Load trained model and corresponding policy
    data = torch.load(model_path)
    policy = data['evaluation/policy']

    if printout:
        print("Policy loaded")
    # Load controller
    controller = env_config.pop("controller")
    if controller in set(ALL_CONTROLLERS):
        # This is a default controller
        controller_config = load_controller_config(default_controller=controller)
    else:
        # This is a string to the custom controller
        controller_config = load_controller_config(custom_fpath=controller)

    # Create robosuite env
    env = suite.make(**env_config,
                     has_renderer=False,
                     has_offscreen_renderer=False,
                     use_object_obs=True,
                     use_camera_obs=False,
                     reward_shaping=True,
                     controller_configs=controller_config
                     )
    env = GymWrapper(env)

    # Use CUDA if available
    if torch.cuda.is_available():
        set_gpu_mode(True)
        policy.cuda() if not isinstance(policy, MakeDeterministic) else policy.stochastic_policy.cuda()

    if printout:
        print("Evaluating policy over {} simulations...".format(n_eval))

    # Create variable to hold rewards to be averaged
    returns = []

    # Loop through simulation n_eval times and take average returns each time
    for i in range(n_eval):
        path = rollout(
            env,
            policy,
            max_path_length=env_config["horizon"],
            render=False,
        )

        # Determine total summed rewards from episode and append to 'returns'
        returns.append(sum(path["rewards"]))

    # Average the episode rewards and return the normalized corresponding value
    return sum(returns) / (env_config["reward_scale"] * n_eval)


def simulate_policy(
        env,
        model_path,
        horizon,
        render=False,
        video_writer=None,
        num_episodes=np.inf,
        printout=False,
        use_gpu=False):
    if printout:
        print("Loading policy...")

    # Load trained model and corresponding policy
    map_location = torch.device("cuda") if use_gpu else torch.device("cpu")
    data = torch.load(model_path, map_location=map_location)
    policy = data['evaluation/policy']

    if printout:
        print("Policy loaded")

    # Use CUDA if available
    if torch.cuda.is_available():
        set_gpu_mode(True)
        policy.cuda() if not isinstance(policy, MakeDeterministic) else policy.stochastic_policy.cuda()

    if printout:
        print("Simulating policy...")

    # Create var to denote how many episodes we're at
    ep = 0

    # Loop through simulation rollouts
    while ep < num_episodes:
        if printout:
            print("Rollout episode {}".format(ep))
        path = rollout(
            env,
            policy,
            max_path_length=horizon,
            render=render,
            video_writer=video_writer,
        )

        # Log diagnostics if supported by env
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()

        # Increment episode count
        ep += 1
