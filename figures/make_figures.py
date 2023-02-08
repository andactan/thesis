import pandas
import os
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

BASE_PATH = os.path.join(os.path.dirname(__file__))

def plot_with_fill(axis, x, y, label, color, type_='line'):
    std = np.std(y, axis=0)
    mean = np.mean(y, axis=0)
    axis.plot(x, mean, label=label, color=color)
    axis.fill_between(x, mean - std / 2, mean + std / 2, color=color, alpha=0.2)


def plot(plots,  xlabel, ylabel, type_='line', name='default'):
    fig, ax = plt.subplots()
    if type_ == 'line':
        for experts, values in plots.items():
            label = experts if isinstance(experts, str) else f"{experts} Experts"
            plot_with_fill(ax, values["x"], values["y"], label, values["color"])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if name != 'Accuracy':
            ax.set_title(name)
        ax.set_xlim([0, int(5e8)])
        ax.set_ylim([-0.1, 1.1])
        ax.grid(True)
        ax.legend()

        fig.savefig(f"{name}")
    
    if type_ == 'bar':
        bars = []
        for expert, values in plots.items():
            bar = ax.bar(str(expert), values['y'], color=values['color'])
            bars.append(bar)
            ax.set_ylim([0, 1.0])
            ax.grid(True, axis='y', alpha=0.2)

        ax.set_title(name)
        ax.set_xlabel('Number of Experts')
        fig.savefig(f"{name}-max")

        
    plt.close(fig)


experts = {
    4: [
        os.path.join(BASE_PATH, "moe-4-0"),
        os.path.join(BASE_PATH, "moe-4-1"),
        os.path.join(BASE_PATH, "moe-4-2"),
    ],
    8: [os.path.join(BASE_PATH, "moe-8-0"), os.path.join(BASE_PATH, "moe-8-1"), os.path.join(BASE_PATH, "moe-8-2")],
    16: [os.path.join(BASE_PATH, "moe-16-0"), os.path.join(BASE_PATH, "moe-16-1")],
    'Base': [os.path.join(BASE_PATH, 'base-0'), os.path.join(BASE_PATH, 'base-1')]
}

if __name__ == "__main__":
    colors = ["red", "blue", 'green']
    environs = {
        "train": [
            "reach",
            "push",
            "pickplace",
            "dooropen",
            "drawerclose",
            "buttonpresstopdown",
            "windowopen",
            "basketball",
        ],
        "test": ["draweropen", "doorclose", "shelfplace", "sweepinto", "leverpull"],
    }

    plots = {}
    for idx, (experts, experiments) in enumerate(experts.items()):

        train_acc_stack = []
        test_acc_stack = []
        environ_acc_stack = defaultdict(lambda: defaultdict(list))
        x = None
        for e in experiments:
            csv = pandas.read_csv(os.path.join(e, "progress.csv"))
            train_acc_stack.append(csv["Traj_Infos/training_episode_success"])
            test_acc_stack.append(csv["Traj_Infos/testing_episode_success"])
            x = csv["Diagnostics/CumSteps"]

            for type, environ_list in environs.items():
                for environ in environ_list:
                    environ_acc_stack[type][environ].append(
                        csv[f"Traj_Infos/{environ}v1_episode_success"]
                    )

        plots[experts] = {
            "x": x,
            "train_acc": np.stack(train_acc_stack),
            "test_acc": np.stack(test_acc_stack),
            "environ_acc_stack": environ_acc_stack,
            "color": colors[idx],
        }

    # training accuracy
    p = {
        expert: {"x": values["x"], "y": values["train_acc"], "color": values["color"]}
        for expert, values in plots.items()
    }

    plot(p, xlabel="Steps", ylabel="Accuracy", name='Training Accuracy')

    # testing accuracy
    p = {
        expert: {"x": values["x"], "y": values["test_acc"], "color": values["color"]}
        for expert, values in plots.items()
    }

    plot(p, xlabel="Steps", ylabel="Accuracy", name='Test Acccuracy')

    # separate environments
    for type, environ_list in environs.items():
        for e in environ_list:
            p = {}
            for expert, values in plots.items():
                p[expert] = {
                    "x": values["x"],
                    "y": np.stack(values["environ_acc_stack"][type][e]),
                    "color": values["color"],
                    "name": e
                }

            plot(p, xlabel="Steps", ylabel="Accuracy", name=e)

    # separate environments
    for type, environ_list in environs.items():
        for e in environ_list:
            p = {}
            for expert, values in plots.items():
                p[expert] = {
                    "y": np.nanmax(np.concatenate(values["environ_acc_stack"][type][e], axis=None)),
                    "color": values["color"]
                }

            plot(p, xlabel="Steps", ylabel="Accuracy", type_='bar', name=e)

    from collections import defaultdict
    def plot_horizontal_bar(plots, label):
        fig, ax = plt.subplots(figsize=(16, 20))

        bars = defaultdict(list)
        colors = []
        for env, expert_dict in plots.items():
            for expert, values in expert_dict.items():
                bars[expert].append(values['y'])
                colors.append(values['color'])

        width = 0.2
        ind = np.arange(len(plots))
        y_ticks = plots.keys()
        x_ticks = [f'{x:.1f}' for x in np.arange(0, 1.2, 0.2)]
        for idx, (expert, expert_list) in enumerate(bars.items()):
            label = expert if isinstance(expert, str) else f'{expert} Experts'
            ax.barh(ind - idx * width, expert_list, width, align='center', label=label, color=colors[idx])

        ax.set_yticks(ind - width)
        ax.set_yticklabels(y_ticks, rotation=55, fontsize=20)
        ax.set_xticklabels(x_ticks, fontsize=17)
        ax.legend(bbox_to_anchor=(0., 1.01, 2., .4), loc='lower left', ncol=3, mode='spread', prop={'size': 20})
        fig.savefig(f'{env_type}-bar-all')

        plt.close(fig)

    
    for env_type, env_list in environs.items():
        p = defaultdict(dict)
        for env in env_list:
            for expert, values in plots.items():
                p[env][expert] = {
                    "y": np.nanmax(np.concatenate(values["environ_acc_stack"][env_type][env], axis=None)),
                    "color": values["color"]
                }

        plot_horizontal_bar(p, label=env_type)

