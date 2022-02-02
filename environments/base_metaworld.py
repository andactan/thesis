import gym
import time
import mujoco_py
import numpy as np

from benchmarks import BENCHMARKS
from policies import POLICIES


class BaseMetaworld(gym.Env):
    def __init__(self, benchmark='ml10', action_repeat=1, demonstration_action_repeat=3, max_trials_per_episode=1,
                 dense_rewards=None, prev_action_observable=False, demonstrations=True, partially_observable=True, visual_observations=False, v2=False, **kwargs) -> None:

        # choose appropriate benchmark and initialize the environment
        self.benchmark = BENCHMARKS[benchmark](**kwargs)

        print(self.benchmark)

        # define action space and applicable arguments
        self.action_space = gym.spaces.Box(low=np.array(
            [-1, -1, -1, -1]), high=np.array([1, 1, 1, 1]))
        self.action_repeat = action_repeat
        self.demonstration_action_repeat = demonstration_action_repeat
        self.max_trials_per_episode = max_trials_per_episode

        # flags
        self.v2 = v2
        self.demonstrations = demonstrations
        self.partially_observable = partially_observable
        self.visual_observations = visual_observations
        self.prev_action_observable = prev_action_observable
        self.demonstration_phase = self.demonstrations

        # demonstration and trial phases
        self.trial_success = False  # if trial was successful
        self.demonstration_success = False  # if demonstration was successful
        self.steps_in_trial = 0  # number of steps spent in a trial phase
        self.steps_in_demonstration = 0  # number of steps spent in demonstration phase
        self.num_trials_in_episode = 0  # number of trials in the episode
        self.num_demons_in_episode = 0  # number of demonstrations in the episode
        self.num_trial_success = 0  # number of successful trials in the episode
        self.num_demon_success = 0  # number of successful demons in the episode

        # compute the observation size
        observation_size = 15  # default value
        if self.prev_action_observable:
            observation_size += np.prod(self.action_space.shape)

        # define observation space
        if self.visual_observations:
            self.observation_space = gym.spaces.Dict({
                'camera_image': gym.spaces.Box(low=-1, high=1, shape=(3, 64, 64)),
                'state': gym.spaces.Box(low=-10, high=10, shape=(observation_size,))
            })

        else:
            self.observation_space = gym.spaces.Dict({
                'state': gym.spaces.Box(low=-10, high=10, shape=(observation_size,)),
                'task_id': gym.spaces.Discrete(50),
                'demonstration_phase': gym.spaces.Discrete(2),
            })

        # misc
        self.dense_rewards = dense_rewards
        self.oracle_policy = None

    def reset(self):
        '''Resets the environment to its default configuration
        '''
        self._reset_env(new_episode=True)
        self.oracle_policy = POLICIES[self.env_name]() if not self.v2 else None

        # reset counters
        self.num_trials_in_episode = 0
        self.num_demons_in_episode = 0
        self.num_trial_success = 0
        self.num_demon_success = 0

        # reset to demonstration phase
        self.demonstration_phase = self.demonstrations

        return self.get_full_observation(self.observation)

    def step(self, action):
        pass

    def get_full_observation(self, observation=None):
        pass

    def setup_camera(self):
        '''Setups the camera
        '''
        if not hasattr(self.env, 'viewer') or self.env.viewer is None:
            viewer = mujoco_py.MjRenderContextOffscreen(self.env.sim, 0)
            viewer.cam.distance = 1
            viewer.cam.fixedcamid = 0
            viewer.cam.azimuth = 130.0
            viewer.cam.elevation = -50
            viewer.cam.lookat[0] = 0
            viewer.cam.lookat[1] = 0.7
            viewer.cam.lookat[2] = 0
            self.env.viewer = viewer

    def _trial_timeout(self):
        return self.steps_in_trial >= self.env.max_path.length / 2

    def _reset_env(self, new_episode=True):
        '''Helper function to reset the environment
        '''
        if new_episode:
            self.env_name, self.env, task, self.env_id = self.benchmark.sample_env_and_task()
            self.env.set_task(task)

            if self.visual_observations:
                self.setup_camera()

        self.observation = self.env.reset()

        # reset counters and flags
        self.trial_success = False
        self.demonstration_success = False
        self.steps_in_trial = 0
        self.steps_in_demonstration = 0


if __name__ == '__main__':
    env = BaseMetaworld(benchmark='ml10', action_repeat=1, demonstration_action_repeat=1,
                               max_trials_per_episode=1, sample_num_classes=1, mode='meta-training', v2=True)

    print(env.reset())
