import torch

import torch_fpsample

x = torch.rand(64, 2048, 3)
# random sample
sampled_points, indices = torch_fpsample.sample(x, 1024)
# random sample with specific tree height
sampled_points, indices = torch_fpsample.sample(x, 1024, h=5)
# random sample with start point index (int)
sampled_points, indices = torch_fpsample.sample(x, 1024, start_idx=0)


