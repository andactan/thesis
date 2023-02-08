# Language-conditioned Meta-Reinforcement Learning for Multi Manipulation Tasks

This repository contains the official code of master's thesis work "Language-conditioned Meta-Reinforcement Learning for Multi Manipulation Tasks".

In order to repeat the experiments, run the script as:
```bash
python main_async.py
```

This will execute the experiment with a random seed, selected by the `rlpyt`. In order to replicate the results presented in the work,

```bash
python main_async.py --seed 0
python main_async.py --seed 1
python main_async.py --seed 2
```

## Creating figures
To create the figures in the thesis work, you can run the `make_figures.py` script in the `/figures` directory as

```bash
python make_figures.py
```

This will create all the figures and store them inside the `/figures` directory.
