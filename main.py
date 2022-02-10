import importlib
import yaml
import os

with open(os.path.join(os.getcwd(), "experiment.yaml"), "r") as file_:
    config = yaml.load(file_, Loader=yaml.Loader)

    # sampler_cls = config["sampler"]["class"]

    # if "async" in sampler_cls.lower():
    #     SamplerCls = getattr(
    #         importlib.import_module("rlpyt.samplers.async_.cpu_sampler"), sampler_cls
    #     )

    # else:
    #     SamplerCls = getattr(
    #         importlib.import_module("rlpyt.samplers.parallel.cpu.sampler"), sampler_cls
    #     )

    # # load runner class
    # runner_cls = config['runner']['class']

    # if 'async' in runner_cls.lower():
    #     RunnerCls = getattr(importlib.import_module('rlpyt.runners.async_rl'), runner_cls)

    # else:
    #     RunnerCls = getattr(importlib.import_module('rlpyt.runners.minibatch_rl'), runner_cls)

    # # load algorithm
    # algo_cls = config['algorithm']['class']
    # algo_kwargs = config['algorithm']['args']
    # AlgoCls = getattr(importlib.import_module(f'algorithms.{algo_cls.lower()}'), algo_cls)
    
    # # load agent class
    # agent_cls = config['agent']['class']
    # agent, prefix = agent_cls.lower().split('agent')[0], 'agent'
    # AgentCls = getattr(importlib.import_module(f'agents.{agent}_{prefix}'), agent_cls)

    # # load agent model
    # agent_model_cls = config['agent']['args']['model']['class']
    # ModelCls = getattr(importlib.import_module('models'), agent_model_cls)
    # models_kwargs = config['agent']['args']['model']['args']
    print(config)