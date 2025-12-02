"""
Custom MuJoCo environment registrations for cross-morphology experiments.
This module registers custom variants of standard MuJoCo environments.
"""
import gymnasium
from gymnasium.envs.registration import register
import numpy as np

# Register Reacher variants with different numbers of joints
# Note: These use the same underlying environment but with different IDs
# In a full implementation, these would have different MuJoCo XML files
# For now, we alias them to the standard Reacher-v5

register(
    id='Reacher-3joints-v0',
    entry_point='gymnasium.envs.mujoco.reacher_v5:ReacherEnv',
    max_episode_steps=50
)

register(
    id='Reacher-4joints-v0',
    entry_point='gymnasium.envs.mujoco.reacher_v5:ReacherEnv',
    max_episode_steps=50
)

register(
    id='Reacher-5joints-v0',
    entry_point='gymnasium.envs.mujoco.reacher_v5:ReacherEnv',
    max_episode_steps=50
)

register(
    id='Reacher-6joints-v0',
    entry_point='gymnasium.envs.mujoco.reacher_v5:ReacherEnv',
    max_episode_steps=50
)

# Register HalfCheetah variants
register(
    id='HalfCheetah-3legs-v0',
    entry_point='gymnasium.envs.mujoco.half_cheetah_v4:HalfCheetahEnv',
    max_episode_steps=1000
)

register(
    id='HalfCheetah-Stand-v0',
    entry_point='gymnasium.envs.mujoco.half_cheetah_v4:HalfCheetahEnv',
    max_episode_steps=1000
)

# Register Walker2d variant
register(
    id='Walker-head-v0',
    entry_point='gymnasium.envs.mujoco.walker2d_v4:Walker2dEnv',
    max_episode_steps=1000
)

print("Custom MuJoCo environments registered successfully")
