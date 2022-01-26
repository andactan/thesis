import random
import copy
from metaworld import Benchmark, _env_dict, _make_tasks, _ML_OVERRIDE
from abc import abstractmethod


MEDIUM_MODE_CLS_DICT = copy.deepcopy(_env_dict.MEDIUM_MODE_CLS_DICT)
MEDIUM_MODE_CLS_DICT['train'].pop('sweep-v1')  # reward function not working
MEDIUM_MODE_CLS_DICT['train'].pop('peg-insert-side-v1')  # demonstration not working
# MEDIUM_MODE_CLS_DICT['test'].pop('lever-pull-v1') #demonstration not working

HARD_MODE_CLS_DICT = copy.deepcopy(_env_dict.HARD_MODE_CLS_DICT)
# HARD_MODE_CLS_DICT['train']['handle-press-side-v1'] = SawyerHandlePressSideEnvV2
HARD_MODE_CLS_DICT['train'].pop('sweep-v1')  # reward function not working
# demonstration policy not working
HARD_MODE_CLS_DICT['train'].pop('peg-insert-side-v1')
HARD_MODE_CLS_DICT['train'].pop('lever-pull-v1')  # reward function not working
HARD_MODE_CLS_DICT['test'].pop('bin-picking-v1')  # reward function not working
# HARD_MODE_CLS_DICT['test'].pop('hand-insert-v1') #reward function not working


class DemonstrationBenchmark(Benchmark):
    def __init__(self, mode='meta-training', sample_num_classes=1):

        # only acceptable modes are: meta-training, meta-testing, and all
        assert mode in ['meta-training', 'meta-testing', 'all']

        super().__init__()

        # classes
        _train_classes = copy.deepcopy(self.TRAIN_CLASSES)
        _test_classes = copy.deepcopy(self.TEST_CLASSES)

        # kwargs
        _train_kwargs = self.train_kwargs
        _test_kwargs = self.test_kwargs

        if mode == 'meta-training':
            classes = _train_classes
            kwargs = _train_kwargs

        elif mode == 'meta-testing':
            classes = _test_classes
            kwargs = _test_kwargs

        else:
            classes = {**_train_classes, **_test_classes}
            kwargs = {**_train_kwargs, **_test_kwargs}

        # sample benchmark classes
        _sample_num_classes = min(len(classes.keys()), sample_num_classes)
        self.classes = dict(random.sample(
            classes.items(), _sample_num_classes))

        # all possible classes included in benchmark
        self.all_classes = {**_train_classes, **_test_classes}

        # kwargs
        self.kwargs = dict((key, value)
                           for key, value in kwargs.items() if key in self.classes)

        # class ids
        self.class_idx = {class_[0]: i for i,
                          class_ in enumerate(self.all_classes.items())}

        # environments
        self._envs = [(name, class_())
                      for name, class_ in self.classes.items()]

        # tasks
        self._tasks = _make_tasks(self.classes,
                                  self.kwargs,
                                  _ML_OVERRIDE)

    def sample_env_and_task(self):
        # randomly choose an environment
        env = random.choice(self._envs)
        env_name, env = env
        global_env_id = self.class_idx[env_name]

        task = random.choice(
            [task for task in self._tasks if task.env_name == env_name])
        return env_name, env, task, global_env_id

    @property
    @abstractmethod
    def TRAIN_CLASSES(self) -> 'OrderedDict[EnvName, Type]':
        """Get all of the environment classes used for training."""
        pass

    @property
    @abstractmethod
    def TEST_CLASSES(self) -> 'OrderedDict[EnvName, Type]':
        """Get all of the environment classes used for testing."""
        pass

    @property
    @abstractmethod
    def train_kwargs(self):
        pass

    @property
    @abstractmethod
    def test_kwargs(self):
        pass


class SingleEnv(DemonstrationBenchmark):
    # train_kwargs = _env_dict.medium_mode_train_args_kwargs
    # test_kwargs = _env_dict.medium_mode_test_args_kwargs
    train_kwargs = _env_dict.HARD_MODE_ARGS_KWARGS['train']
    test_kwargs = _env_dict.HARD_MODE_ARGS_KWARGS['test']

    def __init__(self, env_name, *args, **kwargs):
        self.env_name = env_name
        super().__init__(*args, **kwargs)

    @property
    def TRAIN_CLASSES(self) -> 'OrderedDict[EnvName, Type]':
        classes = dict((name, env) for name, env in _env_dict.HARD_MODE_CLS_DICT['train'].items() if
                       name == self.env_name)
        return classes

    @property
    def TEST_CLASSES(self) -> 'OrderedDict[EnvName, Type]':
        # mode = 'train' if self.meta_training else 'test'
        classes = dict((name, env) for name, env in _env_dict.HARD_MODE_CLS_DICT['test'].items() if
                       name == self.env_name)
        return classes


class ML1(DemonstrationBenchmark):
    train_kwargs = _env_dict.medium_mode_train_args_kwargs
    test_kwargs = _env_dict.medium_mode_test_args_kwargs

    @property
    def TRAIN_CLASSES(self) -> 'OrderedDict[EnvName, Type]':
        classes = dict((name, env) for name, env in _env_dict.EASY_MODE_CLS_DICT.items() if
                       name in ['reach-v1', 'push-v1', 'pick-place-v1'])
        return classes

    @property
    def TEST_CLASSES(self) -> 'OrderedDict[EnvName, Type]':
        return self.TRAIN_CLASSES


class ML10(DemonstrationBenchmark):
    TRAIN_CLASSES = MEDIUM_MODE_CLS_DICT['train']
    TEST_CLASSES = MEDIUM_MODE_CLS_DICT['test']
    train_kwargs = _env_dict.medium_mode_train_args_kwargs
    test_kwargs = _env_dict.medium_mode_test_args_kwargs


class ML45(DemonstrationBenchmark):
    TRAIN_CLASSES = HARD_MODE_CLS_DICT['train']
    TEST_CLASSES = HARD_MODE_CLS_DICT['test']
    train_kwargs = _env_dict.HARD_MODE_ARGS_KWARGS['train']
    test_kwargs = _env_dict.HARD_MODE_ARGS_KWARGS['test']


BENCHMARKS = {
    'ml1': ML1,
    'ml10': ML10,
    'ml45': ML45
}


if __name__ == '__main__':
    d = ML10()
    # print(d._envs)
    print(d.train_kwargs)
