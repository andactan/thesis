import gym
import mujoco_py
import numpy as np
import sys
import os
import cv2
import torch

from base_metaworld import BaseMetaworld
from policies import POLICIES

sys.path.append(os.path.abspath('.'))
print(sys.path)
from models.vision_transformer.model import ViT
from models.pixl2r.model import ImageEncoder


class VisionMetaworld(BaseMetaworld):
    def __init__(
        self,
        benchmark="ml10",
        action_repeat=1,
        demonstration_action_repeat=3,
        max_trials_per_episode=1,
        dense_rewards=None,
        prev_action_observable=False,
        demonstrations=True,
        partially_observable=True,
        visual_observations=False,
        v2=False,
        **kwargs
    ) -> None:
        super().__init__(
            benchmark,
            action_repeat,
            demonstration_action_repeat,
            max_trials_per_episode,
            dense_rewards,
            prev_action_observable,
            demonstrations,
            partially_observable,
            visual_observations,
            v2,
            **kwargs
        )

        self.width = 50
        self.height = 50
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, self.width, self.height), dtype=np.uint8
        )

        # vision encoder
        self.encoder = ImageEncoder()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        return (obs, self._get_observation())

    def reset(self):
        self.env_name, self.env, task, self.env_id = self.benchmark.sample_env_and_task()
        self.oracle_policy = POLICIES[self.env_name]() if not self.v2 else None
        self.env.set_task(task)
        
        return self.env.reset()

    def _get_observation(self):
        frames = self._get_frame()
        for pos, frame in frames.items():
            print(frame.dtype)
            temp = torch.from_numpy(frame.copy())
            temp = temp.view(temp.shape[-1], *temp.shape[:-1]) # batch first
            frames[pos] = temp[None, :]
            print('anan', frames[pos].shape)

        encoded_obs = self.encoder(frames['left'], frames['center'], frames['right'], 1)
        
        return encoded_obs

    def _get_frame(self):
        # setup the camera for 3 different angles and snapshot renders
        obs_dict = {}
        for camera_pos in ["left", "center", "right"]:
            self.setup_camera_(camera_pos)
            self.env.viewer.render(width=self.width, height=self.height)
            obs = self.env.viewer.read_pixels(self.width, self.height, depth=False)
            obs_dict[camera_pos] = obs[::-1, :, :].astype(np.float32) # revert the image

        return obs_dict

    def setup_camera_(self, camera_pos="center"):
        """Setups the camera"""
        
        viewer = None
        if not hasattr(self.env, "viewer") or self.env.viewer is None:
            viewer = mujoco_py.MjRenderContextOffscreen(self.env.sim, device_id=1)

        else:
            viewer = self.env.viewer

        # viewer.cam.lookat = [0, 0.75, 0.4]
        viewer.cam.lookat[0] = 0
        viewer.cam.lookat[1] = 0.75
        viewer.cam.lookat[2] = 0.4
        viewer.cam.elevation = -30  # default elevation
        viewer.cam.trackbodyid = -1
        viewer.cam.distance = 2.0  # default value for left and right

        properties = {
            "center": {"distance": 1.10, "azimuth": 90},
            "left": {"azimuth": 45},
            "right": {"azimuth": 135},
        }

        # set viewer.cam properties
        for key, val in properties[camera_pos].items():
            setattr(viewer.cam, key, val)

        # set the new viewer
        self.env.viewer = viewer


if __name__ == "__main__":
    from PIL import Image
    import os
    # os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
    # os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
    # os.environ['DISPLAY'] = ':0'

    env = VisionMetaworld(
        benchmark="ml10",
        action_repeat=1,
        demonstration_action_repeat=1,
        max_trials_per_episode=1,
        sample_num_classes=1,
        mode="meta-training",
        v2=False,
        visual_observations=True
    )

    obs = env.reset()
    env.env.sim.render(30, 30, device_id=1)
    env._get_observation()
    # for i in range(1000):
    #     action = env.oracle_policy.get_action(obs)
    #     obs, visual = env.step(action)
    #     # env.render()

    #     for pos in ['left', 'center', 'right']:
    #         arr = visual[pos]
    #         im = Image.fromarray(arr)
    #         im.save(f'image_{pos}.png')

        

