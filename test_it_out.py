from environments import LanguageMetaworld
from agents.vmpo_agent import VMPOAgent
from models.vmpo_model import VMPOModel
from misc import CustomEnvInfoWrapper
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.utils.buffer import torchify_buffer
from rlpyt.agents.base import AgentInputs
from rlpyt.envs.base import EnvSpaces

import pickle
import os
import torch


def metaworld_env_wrapper(env):
    info_example = {"timeout": 0}

    return GymEnvWrapper(CustomEnvInfoWrapper(env, info_example))

handle = open(os.path.join('run_250922-130944', 'params.pkl'), 'rb')
config = torch.load(handle)


agent_config = dict(
    ModelCls=VMPOModel,
    model_kwargs=dict(
        sequence_len=64,
        size='medium',
        linear_value_output=False
    )
)

env = LanguageMetaworld(benchmark='ml10', action_repeat=1, demonstration_action_repeat=5,
                        max_trials_per_episode=3, sample_num_classes=5, mode='meta-training')
env = metaworld_env_wrapper(env)

env_spaces = EnvSpaces(
        observation=env.observation_space,
        action=env.action_space,
)

initial_agent_state_dict = config['agent_state_dict']
agent = VMPOAgent(initial_model_state_dict=initial_agent_state_dict, **agent_config)
agent.initialize(env_spaces)



obs = env.reset()
prev_action = torch.tensor(0.0, dtype=torch.float) #None
prev_reward = torch.tensor(0.0, dtype=torch.float) #None

for i in range(100):
    agent_inputs = torchify_buffer(AgentInputs(obs, prev_action, prev_reward))
    agent_step = agent.step(*agent_inputs)
    action = agent_step.action
    obs, reward, done, info = env.step(action)

    prev_action = torch.tensor(action, dtype=torch.float)
    prev_reward = torch.tensor(reward, dtype=torch.float)