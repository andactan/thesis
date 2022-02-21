import importlib
import yaml
import os
import multiprocessing
import time

from datetime import datetime
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.utils.logging.context import logger_context
from rlpyt.envs.gym import GymEnvWrapper
from environments import LanguageMetaworld
from misc import CustomEnvInfoWrapper
from misc.traj_infos import EnvInfoTrajInfo


def metaworld_env_wrapper(**kwargs):
    info_example = {"timeout": 0}
    env = LanguageMetaworld(**kwargs)

    return GymEnvWrapper(CustomEnvInfoWrapper(env, info_example))


def read_config(filename):
    with open(os.path.join(os.getcwd(), filename), "r") as file_:
        config = yaml.load(file_, Loader=yaml.Loader)

        return config


def set_experiment(config, log_dir):
    # load sampler class
    sampler_cls = config["sampler"]["class"]

    if "async" in sampler_cls.lower():
        SamplerCls = getattr(
            importlib.import_module("rlpyt.samplers.async_.cpu_sampler"), sampler_cls
        )

    else:
        SamplerCls = getattr(
            importlib.import_module("rlpyt.samplers.parallel.cpu.sampler"), sampler_cls
        )

    sampler_cls_kwargs = config["sampler"]["args"]
    sampler_cls_kwargs["TrajInfoCls"] = EnvInfoTrajInfo
    sampler = SamplerCls(**sampler_cls_kwargs, EnvCls=metaworld_env_wrapper)

    # load algorithm
    algo_cls = config["algorithm"]["class"]
    algo_kwargs = config["algorithm"]["args"]
    AlgoCls = getattr(importlib.import_module(f"algorithms.{algo_cls.lower()}"), algo_cls)
    algorithm = AlgoCls(**algo_kwargs)

    # load agent class
    agent_cls = config["agent"]["class"]
    _agent_kwargs = config["agent"]["args"]
    agent_kwargs = {
        "ModelCls": getattr(importlib.import_module("models"), _agent_kwargs['model']['class']),
        "model_kwargs": _agent_kwargs["model"]["args"],
    }
    agent_model, prefix = agent_cls.lower().split("agent")[0], "agent"
    AgentCls = getattr(importlib.import_module(f"agents.{agent_model}_{prefix}"), agent_cls)
    agent = AgentCls(**agent_kwargs)

    # load runner class
    runner_cls = config["runner"]["class"]

    if "async" in runner_cls.lower():
        RunnerCls = getattr(importlib.import_module("rlpyt.runners.async_rl"), runner_cls)

    else:
        RunnerCls = getattr(importlib.import_module("rlpyt.runners.minibatch_rl"), runner_cls)

    runner_cls_kwargs = config["runner"]["args"]
    runner_affinity_kwargs = config["runner"]["affinity"]
    runner_affinity_kwargs["n_cpu_core"] = int(
        multiprocessing.cpu_count() * runner_affinity_kwargs["n_cpu_core"]
    )

    # cast float values to int
    runner_cls_kwargs["n_steps"] = int(runner_cls_kwargs["n_steps"])
    runner_cls_kwargs["log_interval_steps"] = int(runner_cls_kwargs["log_interval_steps"])
    affinity = make_affinity(**runner_affinity_kwargs)
    runner = RunnerCls(
        algo=algorithm, agent=agent, sampler=sampler, affinity=affinity, **runner_cls_kwargs
    )

    run_id =  datetime.now().strftime('%d%m%y-%H%M%S')
    name = f'{sampler_cls}-{runner_cls}-{algo_cls}-{agent_cls}-{_agent_kwargs["model"]["class"]}'

    # point the log directory explicitly, otherwise it is gonna be saved in rlpyt itself
    log_dir = os.path.join(os.getcwd(), log_dir)
    with logger_context(log_dir, run_id, name, config, snapshot_mode='last'):
        # start training
        runner.train()


if __name__ == "__main__":
    config = read_config("experiment.yaml")
    set_experiment(config=config, log_dir='experiments')

    # from datetime import datetime
    # print(datetime.now().strftime('%d%m%y-%H%M%S'))

    # gg = 10
    # for i in range(100):
    #     print(i)
    