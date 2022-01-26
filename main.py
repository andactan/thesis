import yaml


with open('experiment.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    print(data)