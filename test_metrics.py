

import torch
torch.manual_seed(8)

batch_size = 4
num_classes = 5

y_pred = torch.rand(batch_size, num_classes)
y = torch.randint(0, num_classes, size=(batch_size, ))