import importlib
import yaml
import os
import multiprocessing
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.envs.gym import GymEnvWrapper
from environments import LanguageMetaworld
from misc import CustomEnvInfoWrapper


# def metaworld_env_wrapper(**kwargs):
#     info_example = {'timeout': 0}
#     env = LanguageMetaworld(**kwargs)

#     return GymEnvWrapper(EnvInfoWrapper(env, info_example))


# def read_config(filename):
#     with open(os.path.join(os.getcwd(), filename), "r") as file_:
#         config = yaml.load(file_, Loader=yaml.Loader)

#         return config


# def set_experiment(config):
#     sampler_cls = config["sampler"]["class"]

#     if "async" in sampler_cls.lower():
#         SamplerCls = getattr(
#             importlib.import_module("rlpyt.samplers.async_.cpu_sampler"), sampler_cls
#         )

#     else:
#         SamplerCls = getattr(
#             importlib.import_module("rlpyt.samplers.parallel.cpu.sampler"), sampler_cls
#         )

#     # load runner class
#     runner_cls = config["runner"]["class"]

#     if "async" in runner_cls.lower():
#         RunnerCls = getattr(importlib.import_module("rlpyt.runners.async_rl"), runner_cls)

#     else:
#         RunnerCls = getattr(importlib.import_module("rlpyt.runners.minibatch_rl"), runner_cls)

#     # load algorithm
#     algo_cls = config["algorithm"]["class"]
#     algo_kwargs = config["algorithm"]["args"]
#     AlgoCls = getattr(importlib.import_module(f"algorithms.{algo_cls.lower()}"), algo_cls)

#     # load agent class
#     agent_cls = config["agent"]["class"]
#     agent, prefix = agent_cls.lower().split("agent")[0], "agent"
#     AgentCls = getattr(importlib.import_module(f"agents.{agent}_{prefix}"), agent_cls)

#     # load agent model
#     agent_model_cls = config["agent"]["args"]["model"]["class"]
#     ModelCls = getattr(importlib.import_module("models"), agent_model_cls)
#     models_kwargs = config["agent"]["args"]["model"]["args"]

#     # load runner
#     runner_cls = config["runner"]["class"]
#     RunnerCls = getattr(
#         importlib.import_module(
#             f'rlpyt.runners.{runner_cls.lower().split("eval")[0].split("rl")[0]}_rl'
#         ),
#         runner_cls,
#     )
#     runner_cls_kwargs = config["runner"]["args"]
#     runner_cls_kwargs["n_cpu_core"] = multiprocessing.cpu_count() * runner_cls_kwargs["n_cpu_core"]
#     affinity = make_affinity(**runner_cls_kwargs)


