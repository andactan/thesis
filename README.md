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
To create the figures in the thesis work, you can run the `make_figures.py` script in the `figures` directory as

```bash
python make_figures.py
```

This will create all the figures and store them inside the `figures` directory.

## Generating Contextual Embeddings
To generate contextual embeddings for the environment name, run `context_embeddings.py` in `environment` directory as

```bash
python context_embeddings.py
```

This will save the embeddings as a `dict` structure in which keys are the environment names and the values are the context embeddings, and it is pickled into the file `context_embeddings_roberta.pkl`.
