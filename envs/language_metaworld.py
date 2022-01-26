import gym
import pickle
import os
import random
import numpy as np

from base_metaworld import BaseMetaworld
from torchtext.vocab import GloVe
from instructions import INSTRUCTIONS


class LanguageMetaworld(BaseMetaworld):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.instruction_idx = 0
        self.instruction = None
        self.embed_dim = 50

        # load vocabulary
        vocab_path = os.path.join(os.path.dirname(
            __file__), 'metaworld_vocab.pkl')
        self.embeddings = pickle.load(os.open(vocab_path, 'rb'))

        # define the observation space
        if self.visual_observations:
            self.observation_space = gym.spaces.Dict({
                'camera_image': gym.spaces.Box(low=-1, high=1, shape=(3, 64, 64)),
                'state': gym.spaces.Box(low=-10, high=10, shape=(self.EMBED_DIM,)),
            })

        else:
            self.observation_space = gym.spaces.Dict({
                'state': gym.spaces.Box(low=-10, high=10, shape=(self.EMBED_DIM,)),
                'task_id': gym.spaces.Discrete(50),
                'demonstration_phase': gym.spaces.Discrete(2),
            })

    def step(self, action):
        reward_sum = 0
        if self.demonstration_phase:
            instruction, instruction_end = self.get_instruction_word()
            full_observation = self.get_full_observation(
                instruction_word=instruction)
            self.demonstration_phase = not instruction_end

        else:
            # trial
            for _ in range(self.action_repeat):
                self.env._partially_observable = True
                self.observation, reward, done, info = self.env.step(action)
                self.steps_in_trial += 1
                self.trial_success = info['success']
                reward_sum += reward

            if self._trial_timeout():
                # if number of steps taken in the trial phase
                # exceeds the allowed number of steps, stop
                self.num_trial_success += self.trial_success
                self.num_trials_in_episode += 1
                # go over demonstration again if failed in trial
                self.demonstration_phase = not self.trial_success
                self._reset_env()  # reset the env without going into a new episode

            full_observation = self.get_full_observation(self.observation)

        # TODO: fill out the info parameter
        return full_observation, reward_sum, done, {}

    def get_instruction_word(self):
        word = self.instruction[self.instruction_idx]
        if word == 'goal_pos':
            embedding = np.concatenate(
                (self.env._get_pos_goal(), np.zeros(self.EMBED_DIM - 3)))
        else:
            embedding = self.embeddings[word]

        # update instruction index and return if processing has finished
        self.instruction_idx = (self.instruction_idx +
                                1) % len(self.instruction)
        instruction_end = self.instruction_idx == 0

        return embedding, instruction_end

    # TODO: implement get_full_observation
    def get_full_observation(self, observation=None):
        pass

    def _reset_env(self, new_episode=True):
        super()._reset_env(new_episode)

        # sample a new instruction
        self.instruction = random.sample(
            INSTRUCTIONS[self.env_name], 1)[0].split()
        self.instruction_idx = 0
