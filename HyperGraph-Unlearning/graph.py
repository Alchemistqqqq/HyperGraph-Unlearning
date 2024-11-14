import torch
from dhg.data import Cooking200

data = Cooking200()

print(data)
print(data['num_classes'])
print(data['num_vertices'])
print(data['num_edges'])
print(data['edge_list'])
print(data['labels'])

print(data['train_mask'])
print(data['val_mask'])
print(data['test_mask'])