import torch

from rlpyt.agents.base import (BaseAgent, RecurrentAgentMixin, AgentStep)
from rlpyt.agents.pg.mujoco import MujocoMixin
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from rlpyt.agents.pg.base import AgentInfoRnn
from rlpyt.utils.buffer import buffer_to, buffer_func, buffer_method
from rlpyt.utils.collections import namedarraytuple

DistInfo = namedarraytuple("DistInfo", ["mean", 'std'])


class VMPOAgent(MujocoMixin, RecurrentAgentMixin, BaseAgent):
    """Reference VMPO implementation for action selection
    """

    def initialize(self, env_spaces, share_memory=False, **kwargs):
        super().initialize(env_spaces, share_memory, **kwargs)

        self.distribution = Gaussian(dim=env_spaces.action.shape[0])

    def __call__(self, observation, previous_action, previous_reward, initial_rnn_state):
        """Performs forward call on training data for algorithm
        """

        # copy model inputs to intended device
        input = buffer_to(
            (observation, previous_action, previous_reward, initial_rnn_state),
            device=self.device)

        output = self.model(*input)
        mu, std, value, rnn_state = output[:4]
        dist_info, value = buffer_to(
            (DistInfoStd(mean=mu, log_std=std), value), device="cpu")

        return dist_info, value, rnn_state, *output[4:]

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        from colorama import Fore
        print(Fore.RED + '--> AGENT STEP')

        agent_inputs = buffer_to(
            (observation, prev_action, prev_reward), device=self.device)

        with torch.cuda.amp.autocast():
            model_output = self.model(*agent_inputs, self.prev_rnn_state)

        mu, std, value, rnn_state = model_output[:4]
        dist_info = DistInfoStd(mean=mu, log_std=std)
        dist = torch.distributions.normal.Normal(loc=mu, scale=std)
        action = dist.sample() if self._mode == 'sample' else mu

        if self.prev_rnn_state is None:
            prev_rnn_state = buffer_func(rnn_state, torch.zeros_like)
        else:
            prev_rnn_state = self.prev_rnn_state

        # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
        # (Special case: model should always leave B dimension in.)
        prev_rnn_state = buffer_method(prev_rnn_state, "transpose", 0, 1)
        agent_info = AgentInfoRnn(dist_info=dist_info, value=value,
                                  prev_rnn_state=prev_rnn_state)
        action, agent_info = buffer_to((action, agent_info), device="cpu")

        self.advance_rnn_state(rnn_state)  # Keep on device.
        
        return AgentStep(action=action, agent_info=agent_info)
