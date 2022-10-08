import seaborn as sns
import matplotlib.pyplot as plt
import pandas
import os


path_baseline = os.path.join(os.getcwd(), "run_160922-151557", "progress.csv")
path_improve = os.path.join(os.getcwd(), 'run_250922-130944', 'progress.csv')
df_baseline = pandas.read_csv(path_baseline, header=0)
df_improve = pandas.read_csv(path_improve, header=0)

columns = [
    "Diagnostics/CumSteps",
    "Traj_Infos/training_episode_success",
    "Traj_Infos/testing_episode_success",
]

# training accuracy
fig, ax = plt.subplots()
ax.set_xlabel('Steps')
ax.set_ylabel('Percentage Success')
sns.lineplot(data=df_baseline[columns], x='Diagnostics/CumSteps', y='Traj_Infos/training_episode_success', ax=ax)
sns.lineplot(data=df_improve[columns], x='Diagnostics/CumSteps', y='Traj_Infos/training_episode_success', ax=ax)
plt.legend(labels=['baseline', 'moe'])
fig.savefig('training_accuracy.png')

# drawer-open-v1 and door-close-v1
for test in ['drawer-open-v1', 'door-close-v1']:
    key = "".join(test.split('-'))
    key = f'Traj_Infos/{key}_episode_success'

    # get 10 largest accuracies
    top_baseline = df_baseline[key].nlargest(20)
    top_improve = df_improve[key].nlargest(20)
    
    labels = ['Baseline', 'Mixture of Experts']
    data = [top_baseline.mean(), top_improve.mean()]

    fig, ax = plt.subplots()
    ax.set_xlabel('Model')
    ax.set_ylabel('Percentage Success')
    ax.set_title(test)

    sns.barplot(x=labels, y=data)

    fig.savefig(f'{test}.png')


training_benchmarks = [
    'reach',
    'push',
    'door-open',
    'drawer-close',
    'window-open'
]

# figure size in inches
fig, ax = plt.subplots(5, 1, figsize=(15, 12))
plt.subplots_adjust(hspace=0.6)
for idx, training_benchmark in enumerate(training_benchmarks):
    key = "".join(training_benchmark.split('-'))
    key = f'Traj_Infos/{key}v1_episode_success'

    ax[idx].set_xlabel('Steps')
    ax[idx].set_ylabel('Percentage Success')
    ax[idx].set_title(key)

    columns = [key, 'Diagnostics/CumSteps']

    sns.lineplot(data=df_baseline[columns], x=columns[1], y=columns[0], ax=ax[idx])
    sns.lineplot(data=df_improve[columns], x=columns[1], y=columns[0], ax=ax[idx])

labels = ['baseline', 'MoE']
fig.legend(labels, loc='lower right')

fig.savefig('training_accuracies.png')
