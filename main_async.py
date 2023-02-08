from datetime import datetime
import multiprocessing
import os
import GPUtil
import sys
import argparse

# sys.path.remove('/data/yaoxt3/RL/rlpyt')
# sys.path.append('/data/yaoxt3/Sina-thesis/rlpyt')

from algorithms.vmpo.vmpo import VMPO
from algorithms.async_vmpo.async_vmpo import AsyncVMPO
from algorithms.async_vmpo.async_vmpo_mixin import MultitaskAsyncVMPO
from environments.language_metaworld import LanguageMetaworld
from misc.traj_infos import EnvInfoTrajInfo
from agents.vmpo_agent import VMPOAgent
from models import VMPOModel
from misc import CustomEnvInfoWrapper

from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler
from rlpyt.samplers.parallel.cpu.collectors import CpuWaitResetCollector
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.utils.logging.context import logger_context


def choose_affinity(slot_affinity_code):
    if slot_affinity_code is None:
        num_cpus = multiprocessing.cpu_count()
        num_gpus = len(GPUtil.getGPUs())

    affinity = make_affinity(
        n_cpu_core=8,
        cpu_per_run=8,
        n_gpu=1,
        async_sample=True,
        optim_sample_share_gpu=False,
        alternating=False,
        set_affinity=True,
        n_socket=1)

    affinity['optimizer'][0]['cuda_idx'] = 0
    affinity['cuda_idx'] = 0

    print(f'Affinity -> {affinity}')
    return affinity

def metaworld_env_wrapper(**kwargs):
    info_example = {"timeout": 0}
    env = LanguageMetaworld(**kwargs)

    return GymEnvWrapper(CustomEnvInfoWrapper(env, info_example))

def build_and_train(slot_affinity_code=None, log_dir='experiments', serial_mode=False, alternating_sampler=False, name='run', seed=None):
    sequence_length = 8 # changed 64 -> 80
    config = dict(
        algo_kwargs=dict(
            epochs=4,
            minibatches=1,
            T_target_steps=100, # changed 100 -> 120
            batch_B=2, # changed 128 -> 256
            batch_T=sequence_length,
            epsilon_eta=0.1,
            gae_lambda=1,
            discrete_actions=False
        ),

        sampler_kwargs=dict(
            batch_T=sequence_length, # number of time steps to be taken in each environment
            batch_B=2, # number of parallel envs
            eval_n_envs=2,
            eval_max_steps=1e5, # changed 
            eval_max_trajectories=400, # changed 360 -> 400
            TrajInfoCls=EnvInfoTrajInfo,
            env_kwargs=dict(
                action_repeat=2, # changed 2 => 4
                demonstration_action_repeat=5, # not used
                max_trials_per_episode=3, # changed 3 -> 4
                mode='meta-training',
                benchmark='ml10'
            ),
            eval_env_kwargs=dict(
                benchmark='ml10',
                action_repeat=2, # changed 2 -> 4
                demonstration_action_repeat=5, # not used
                max_trials_per_episode=3, # changed 3 -> 4
                mode='all'
            )
        ),

        agent_kwargs=dict(
            ModelCls=VMPOModel,
            model_kwargs=dict(
                sequence_len=sequence_length,
                size='medium',
                linear_value_output=False
            )
        ),

        runner_kwargs=dict(
            n_steps=5e8, # changed 5e8 -> 5e7
            log_interval_steps=5e6
        ),
    )

    AlgoCls=AsyncVMPO
    SamplerCls=AsyncCpuSampler
    RunnerCls=AsyncRlEval
    AgentCls=VMPOAgent

    affinity = choose_affinity(
        slot_affinity_code=slot_affinity_code
    )

    sampler = SamplerCls(**config['sampler_kwargs'], EnvCls=metaworld_env_wrapper)
    algorithm = AlgoCls(**config['algo_kwargs'])
    agent = AgentCls(**config['agent_kwargs'])
    runner = RunnerCls(
        **config['runner_kwargs'],
        algo=algorithm,
        agent=agent,
        sampler=sampler,
        # affinity=dict(cuda_idx=0, workers_cpus=list(range(8)))
        affinity=affinity,
        seed=seed
    )

    log_dir = os.path.join(os.getcwd(), log_dir)
    run_id =  datetime.now().strftime('%d%m%y-%H%M%S')
    with logger_context(log_dir, run_id, name, config, snapshot_mode='last'):
        # start training
        runner.train()

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="Seed value")
    args = parse.parse_args()
    seed = args.seed
    
    build_and_train(seed=seed)
