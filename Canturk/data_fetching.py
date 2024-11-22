# pip install torch-geometric

from torch_geometric.datasets import TUDataset

dataset_names = ['NCI1', 'DD', 'PROTEINS', 'MUTAG', 'PTC_FM', 'PTC_MR', 'IMDB-BINARY']

root_dir = 'data'

for name in dataset_names:
    dataset = TUDataset(root=f'{root_dir}/{name}', name=name)
