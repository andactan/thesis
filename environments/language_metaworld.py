import torch
import gym
import pickle
import os
import random
import mujoco_py
import numpy as np

from environments.base_metaworld import BaseMetaworld
from environments.instructions import INSTRUCTIONS

class LanguageMetaworld(BaseMetaworld):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.instruction_idx = 0
        self.instruction = None
        self.embed_dim = 50

        # image properties
        self.width_img = 64
        self.height_img = 64

        # load vocabulary
        vocab_path = os.path.join(os.path.dirname(
            __file__), 'metaworld_vocab.pkl')
        self.embeddings = pickle.load(open(vocab_path, 'rb'))

        # load context embeddings
        context_path = os.path.join(os.path.dirname(__file__), 'context_embeddings_roberta.pkl')
        self.context_embeddings = pickle.load(open(context_path, 'rb'))
        self.context_dim = 768

        # define the observation space
        if self.visual_observations:
            self.observation_space = gym.spaces.Dict({
                'camera_image': gym.spaces.Box(low=-1, high=1, shape=(3, 64, 64)),
                'state': gym.spaces.Box(low=-10, high=10, shape=(self.embed_dim,)),
                'task_id': gym.spaces.Discrete(50),
                'demonstration_phase': gym.spaces.Discrete(2),
            })

        else:
            # add context to observation_space
            self.observation_space = gym.spaces.Dict({
                'state': gym.spaces.Box(low=-10, high=10, shape=(self.embed_dim,)),
                'task_id': gym.spaces.Discrete(50),
                'demonstration_phase': gym.spaces.Discrete(2),
                'context': gym.spaces.Box(low=-10, high=-10, shape=(self.context_dim, ))
            })

    def step(self, action):

        reward_sum = 0
        if self.demonstration_phase:
            instruction, instruction_end = self.get_instruction_word()
            full_observation = self.get_full_observation(
                instruction=instruction)
            self.demonstration_phase = not instruction_end
            info = {}

        else:
            # trial
            for _ in range(self.action_repeat):
                self.env._partially_observable = True
                self.observation, reward, done, info = self.env.step(action)
                self.steps_in_trial += 1
                self.trial_success = self.trial_success or info['success']
                reward_sum += reward

            if self._trial_timeout():
                # if number of steps taken in the trial phase
                # exceeds the allowed number of steps, stop
                self.num_trial_success += self.trial_success
                self.num_trials_in_episode += 1

                # go over demonstration again if failed in trial
                self.demonstration_phase = not self.trial_success
                self._reset_env(new_episode=False)  # reset the env without going into a new episode

            full_observation = self.get_full_observation(self.observation)

        # TODO: fill out the info parameter
        done = self.num_trials_in_episode >= self.max_trials_per_episode
        info = self._append_env_info(info)
        return full_observation, reward_sum, done, info

    def get_instruction_word(self):
        word = self.instruction[self.instruction_idx]
        if word == 'goal_pos':
            embedding = np.concatenate(
                (self.env._get_pos_goal(), np.zeros(self.embed_dim - 3)))
        else:
            embedding = self.embeddings[word]

        # update instruction index and return if processing has finished
        self.instruction_idx = (self.instruction_idx +
                                1) % len(self.instruction)
        instruction_end = self.instruction_idx == 0

        return embedding, instruction_end

    # TODO: implement get_full_observation
    def get_full_observation(self, observation=None, instruction=None):
        # add another observation as context or env_context
        if observation is not None:
            state = np.concatenate((
                observation,
                [
                    self.demonstration_phase, self.steps_in_trial / 150,
                    self.trial_success and not self.demonstration_phase
                ]
            ))
            # fill out the remaining slots
            state = np.concatenate(
                (state, np.zeros(self.embed_dim - state.shape[0])))

        else:
            # instruction is not None
            state = np.array(instruction)

        context_name = self.env_name.replace('-v1', '').replace('-', ' ')
        context = self.context_embeddings[context_name]

        return dict(
            state=state,
            task_id=np.array(self.env_id),
            demonstration_phase=np.array(int(self.demonstration_phase)),
            context=context
        )

    def _reset_env(self, new_episode=True):
        super()._reset_env(new_episode)

        # # setup cameras and define configurations
        # self._setup_camera(camera_pos='left')
        # self._setup_camera(camera_pos='center')
        # self._setup_camera(camera_pos='right')

        # sample a new instruction
        self.instruction = random.sample(
            INSTRUCTIONS[self.env_name], 1)[0].split()
        self.instruction_idx = 0


    # def _get_visual_observations(self, camera_poss=['center']):
        # obs_dict = {}
        # for camera_pos in camera_poss:
        #     with self.mujoco_context_lock:
        #         print(f'{self.env_name} - {self.env_id} got the lock')
        #         # self._setup_camera(camera_pos=camera_pos)
        #         obs = self.env.sim.render(width=self.width_img, height=self.width_img, mode='offscreen')
        #         obs_dict[camera_pos] = obs[::-1, :, :].astype(np.float32) # revert the image

        #         for i in range(len(self.env.sim.render_contexts)):
        #             self.env.sim.render_contexts[i].__dealloc__()

        # print(f'{self.env_name} - {self.env_id} released the lock')
        # return obs_dict
        # with self.mujoco_context_lock:
        #     # print(f'{self.env_name} - {self.env_id} got the lock')
        #     for i in range(len(self.env.sim.render_contexts)):
        #         del self.env.sim.render_contexts[i]

        #     sim = self.env.sim
        #     render_ctx = mujoco_py.MjRenderContext(sim=sim, device_id=0, offscreen=True, opengl_backend='opengl')
            
        #     render_ctx.render(64, 64)
        #     x = render_ctx.read_pixels(64, 64)
        #     del render_ctx
        #     # self.env.close()
        #     # x = gl.glReadPixels(0, 0, 20, 20)
        #     # print(f'{self.env_name} - {self.env_id} released the lock')
            # return dict(center=x)


    # def _setup_camera(self, camera_pos='center'):
    #     """ Setups the camera in 3 different points """
    #     viewer = None
    #     if not hasattr(self.env, "viewer") or self.env.viewer is None:
    #         viewer = mujoco_py.MjRenderContextOffscreen(self.env.sim, device_id=0)
    #         # viewer = mujoco_py.MjRenderContext(sim=self.env.sim, device_id=1, offscreen=True)

    #     else:
    #         viewer = self.env.viewer

    #     viewer.cam.lookat[0] = 0
    #     viewer.cam.lookat[1] = 0.75
    #     viewer.cam.lookat[2] = 0.4
    #     viewer.cam.elevation = -30  # default elevation
    #     viewer.cam.trackbodyid = -1
    #     viewer.cam.distance = 2.0  # default value for left and right

    #     properties = {
    #         "center": {"distance": 1.10, "azimuth": 90},
    #         "left": {"azimuth": 45},
    #         "right": {"azimuth": 135},
    #     }

    #     # set viewer.cam properties
    #     for key, val in properties[camera_pos].items():
    #         setattr(viewer.cam, key, val)

    #     # set the new viewer
    #     self.env.viewer = viewer

if __name__ == '__main__':
    import os

    # os.environ['LD_PRELOAD'] = "/usr/lib/x86_64-linux-gnu/libGLEW.so"

    env = LanguageMetaworld(benchmark='ml10', action_repeat=1, demonstration_action_repeat=5,
                            max_trials_per_episode=3, sample_num_classes=5, mode='meta-training')

    env.reset()


    print(env.embeddings)