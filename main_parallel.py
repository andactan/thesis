from datetime import datetime
import multiprocessing
import os
# import GPUtil
import sys

# os.environ['LD_PRELOAD'] = f"{os.environ['HOME']}/.mujoco/mujoco210/bin/libglewegl.so"
# sys.path.remove('/data/yaoxt3/RL/rlpyt')
# sys.path.append('/data/yaoxt3/Sina-thesis/rlpyt')

from algorithms.vmpo.vmpo import VMPO
from environments.language_metaworld import LanguageMetaworld
from misc.traj_infos import EnvInfoTrajInfo
from agents.vmpo_agent import VMPOAgent
from models import VMPOModel
from misc import CustomEnvInfoWrapper

from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.cpu.collectors import CpuWaitResetCollector
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.utils.logging.context import logger_context


def choose_affinity(slot_affinity_code):
    if slot_affinity_code is None:
        num_cpus = multiprocessing.cpu_count()
        # num_gpus = len(GPUtil.getGPUs())

    affinity = make_affinity(
        n_cpu_core=16,
        n_gpu=1, 
        set_affinity=True)

    print(f'Affinity -> {affinity}')
    return affinity

def metaworld_env_wrapper(**kwargs):
    info_example = {"timeout": 0}
    env = LanguageMetaworld(**kwargs)

    return GymEnvWrapper(CustomEnvInfoWrapper(env, info_example))

def build_and_train(slot_affinity_code=None, log_dir='experiments', serial_mode=False, alternating_sampler=False, name='run'):
    sequence_length = 16
    num_workers_cpus = 20
    mujoco_context_lock = multiprocessing.Lock()
    config = dict(
        algo_kwargs=dict(
            epochs=2,
            minibatches=1,
            pop_art_reward_normalization=True
        ),

        sampler_kwargs=dict(
            batch_T=sequence_length,
            batch_B=2,
            eval_n_envs=1,
            eval_max_steps=1e3,
            eval_max_trajectories=num_workers_cpus * 16,
            TrajInfoCls=EnvInfoTrajInfo,
            env_kwargs=dict(
                mujoco_context_lock=mujoco_context_lock,
                action_repeat=2,
                demonstration_action_repeat=5,
                max_trials_per_episode=3,
                mode='meta-training',
                benchmark='ml10',
            ),
            eval_env_kwargs=dict(
                mujoco_context_lock=mujoco_context_lock,
                benchmark='ml10',
                action_repeat=2,
                demonstration_action_repeat=5,
                max_trials_per_episode=3,
                mode='all',
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
            n_steps=1e5,
            log_interval_steps=5e6
        ),
    )

    AlgoCls=VMPO
    SamplerCls=CpuSampler
    RunnerCls=MinibatchRlEval
    AgentCls=VMPOAgent

    # affinity = choose_affinity(
    #     slot_affinity_code=slot_affinity_code
    # )

    sampler = SamplerCls(**config['sampler_kwargs'], EnvCls=metaworld_env_wrapper)
    algorithm = AlgoCls(**config['algo_kwargs'])
    agent = AgentCls(**config['agent_kwargs'])
    runner = RunnerCls(
        **config['runner_kwargs'],
        algo=algorithm,
        agent=agent,
        sampler=sampler,
        # affinity=dict(cuda_idx=0)
        affinity=dict(cuda_idx=0, workers_cpus=list(range(num_workers_cpus)))
        # affinity=affinity
    )

    log_dir = os.path.join(os.getcwd(), log_dir)
    run_id =  datetime.now().strftime('%d%m%y-%H%M%S')
    with logger_context(log_dir, run_id, name, config, snapshot_mode='last'):
        # start training
        runner.train()

if __name__ == '__main__':
    build_and_train()