import torch

import torch_fpsample

x = torch.rand(64, 2048, 3)
# random sample
sampled_points, indices = torch_fpsample.sample(x, 1024)
# random sample with specific tree height
sampled_points, indices = torch_fpsample.sample(x, 1024, h=5)
# random sample with start point index (int)
sampled_points, indices = torch_fpsample.sample(x, 1024, start_idx=0)

# TorchScript로 모델을 감싸서 export
class FPSModel(torch.nn.Module):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples
    
    def forward(self, x):
        sampled_points, indices = torch_fpsample.sample(x, self.num_samples)
        return sampled_points, indices

# 모델 인스턴스 생성
model = FPSModel(1024)

# TorchScript로 변환
scripted_model = torch.jit.script(model)

# 모델 저장
scripted_model.save("fps_model.pt")

# 저장된 모델 로드 및 테스트
loaded_model = torch.jit.load("fps_model.pt")
test_input = torch.rand(64, 2048, 3)
output_points, output_indices = loaded_model(test_input)
print(f"Output shape: {output_points.shape}, Indices shape: {output_indices.shape}")
