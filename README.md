# Cross Domain Preference Consistency (CDPC)

# CDPC mujoco
### Run the experiment (defualt, mujoco, reacher)
```shell
python mujoco/CDPC.py --source_env Reacher-v4 --target_env Reacher--3joints
```

# CDPC robosuite
### Run the experiment (defualt, robosuite, lift)
```shell
# Download official implementation from https://github.com/ARISE-Initiative/robosuite-benchmark/tree/master and follow the instructions
python robosuite/CDPC.py --env_name_s Lift --robots_s Panda --env_name_t Lift --robots_t IIWA
```
