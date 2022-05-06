import gym
import mujoco_py
import numpy
from base_metaworld import BaseMetaworld
from policies import POLICIES


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

        self.width = 640
        self.height = 480
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, self.width, self.height), dtype=numpy.uint8
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        return (obs, self.get_observation_())

    def reset(self):
        self.env_name, self.env, task, self.env_id = self.benchmark.sample_env_and_task()
        self.oracle_policy = POLICIES[self.env_name]() if not self.v2 else None
        self.env.set_task(task)
        
        return self.env.reset()

    def get_observation_(self):
        return self.get_frame_()

    def get_frame_(self):
        # setup the camera for 3 different angles and snapshot renders
        obs_dict = {}
        for camera_pos in ["left", "center", "right"]:
            self.setup_camera_(camera_pos)
            self.env.viewer.render(width=self.width, height=self.height)
            obs = self.env.viewer.read_pixels(self.width, self.height, depth=False)
            obs_dict[camera_pos] = obs[::-1, :, :] # revert the image

        return obs_dict

    def setup_camera_(self, camera_pos="center"):
        """Setups the camera"""

        print("setupping camera...")
        viewer = None
        if not hasattr(self.env, "viewer") or self.env.viewer is None:
            viewer = mujoco_py.MjRenderContextOffscreen(self.env.sim, -1)

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

        for key, val in properties[camera_pos].items():
            setattr(viewer.cam, key, val)

        # set the new viewer
        self.env.viewer = viewer


if __name__ == "__main__":
    from PIL import Image


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
    for i in range(1000):
        action = env.oracle_policy.get_action(obs)
        obs, visual = env.step(action)
        # env.render()

        for pos in ['left', 'center', 'right']:
            arr = visual[pos]
            im = Image.fromarray(arr)
            im.save(f'image_{pos}.png')

        

