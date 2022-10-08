import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import metaworld
import random
import torch


from agents.vmpo_agent import VMPOAgent
from environments.language_metaworld import LanguageMetaworld


snapshot = torch.load(os.path.join(os.path.dirname(__file__), 'run_250922-130944', 'params.pkl'))
snapshot['agent_state_dict']



# ml10 = metaworld.ML10()  # Construct the benchmark, sampling tasks

# testing_envs = []
# for name, env_cls in ml10.test_classes.items():
#     env = env_cls()
#     task = random.choice([task for task in ml10.test_tasks if task.env_name == name])
#     env.set_task(task)
#     testing_envs.append(env)

# for env in testing_envs:
#     print(env)
#     obs = env.reset()  # Reset environment
#     a = env.action_space.sample()  # Sample an action
#     obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
