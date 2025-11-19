# PyTorch Custom Operations

PyTorch 커스텀 연산 모음: **Farthest Point Sampling (FPS)** 와 **K-Nearest Neighbors (KNN)** 구현.

이 프로젝트는 다음을 포함합니다:
- `torch_fpsample`: 효율적인 최원점 샘플링 (Farthest Point Sampling) 구현
- `knn_torch3d`: K-최근접 이웃 (K-Nearest Neighbors) 연산 및 gather 기능

**해당 연산은 CPU 전용으로 구현되어 있습니다.**

> [!NOTE]
> PyTorch의 **C++ 내부 병렬 처리(Parallel Backend)** 를 활용하여 효율적인 성능을 제공합니다.

---

## 환경 설정

### 1. 가상환경 생성 및 활성화
`scanvision`에 구현된 `libtorch` 및 `CUDA` version에 맞춰 환경을 설정합니다. 

#### Python 가상환경 생성
```bash
conda create -n ops python=3.10
conda activate ops

# CUDA 12.8 버전 
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

---

## 설치

```bash
### GitHub에서 직접 설치
pip install git+https://github.com/yourusername/torchCustomOps

### 로컬 빌드
# 프로젝트 디렉토리에서
pip install -e .
```

## 사용법

### 1. Farthest Point Sampling (FPS)
```python
import torch
import torch_fpsample

# 입력 포인트 클라우드 (batch_size, num_points, dimensions)
x = torch.rand(64, 2048, 3)

# 기본 샘플링 (1024개의 포인트 선택)
sampled_points, indices = torch_fpsample.sample(x, 1024)

# 트리 높이 지정
sampled_points, indices = torch_fpsample.sample(x, 1024, h=5)

# 시작 포인트 인덱스 지정
sampled_points, indices = torch_fpsample.sample(x, 1024, start_idx=0)

print(f"Sampled points shape: {sampled_points.shape}")  # [64, 1024, 3]
print(f"Indices shape: {indices.shape}")                # [64, 1024]
```

### 2. K-Nearest Neighbors (KNN)
```python
import torch
from knn_torch3d import knn_points, knn_gather

# 포인트 클라우드 생성
batch_size = 2
N1 = 100  # query points 개수
N2 = 200  # reference points 개수
D = 3     # 차원 (x, y, z)

p1 = torch.randn(batch_size, N1, D)  # query points
p2 = torch.randn(batch_size, N2, D)  # reference points

# KNN 계산 (K=5개의 최근접 이웃 찾기)
result = knn_points(p1, p2, K=5, return_nn=True)
dists = result.dists  # 거리 [batch_size, N1, K]
idx = result.idx      # 인덱스 [batch_size, N1, K]
knn = result.knn      # 최근접 이웃 좌표 [batch_size, N1, K, D]

# knn_gather를 사용한 이웃 좌표 가져오기
neighbors = knn_gather(p2, idx)  # [batch_size, N1, K, D]

print(f"Distances shape: {dists.shape}")    # [2, 100, 5]
print(f"Indices shape: {idx.shape}")        # [2, 100, 5]
print(f"Neighbors shape: {neighbors.shape}")# [2, 100, 5, 3]### 3. TorchScript Export
```


#### FPS 모델 Export
```python
import torch
import torch_fpsample

class FPSModel(torch.nn.Module):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples
    
    def forward(self, x):
        sampled_points, indices = torch_fpsample.sample(x, self.num_samples)
        return sampled_points, indices

# 모델 변환 및 저장
model = FPSModel(1024)
scripted_model = torch.jit.script(model)
scripted_model.save("fps_model.pt")

# 로드 및 사용
loaded_model = torch.jit.load("fps_model.pt")
output = loaded_model(torch.rand(64, 2048, 3))
```

#### KNN 모델 Export
```python
import torch
from knn_torch3d import knn_points, knn_gather

class KNNModule(torch.nn.Module):
    def __init__(self, K=5):
        super().__init__()
        self.K = K
    
    def forward(self, p1, p2):
        result = knn_points(p1, p2, K=self.K, return_nn=False)
        neighbors = knn_gather(p2, result.idx)
        return result.dists, result.idx, neighbors

# 모델 변환 및 저장
model = KNNModule(K=5)
scripted_model = torch.jit.script(model)
scripted_model.save("knn_model.pt")

# 로드 및 사용
loaded_model = torch.jit.load("knn_model.pt")
p1 = torch.randn(2, 100, 3)
p2 = torch.randn(2, 200, 3)
dists, idx, neighbors = loaded_model(p1, p2)
```

---

## 튜토리얼

프로젝트에는 두 가지 튜토리얼 스크립트가 포함되어 있습니다:

- `fps_tutorial.py`: FPS 사용법 및 TorchScript export 예제
- `knn_tutorial.py`: KNN 사용법, 검증 함수, TorchScript export 예제

```bash
# 튜토리얼 실행
python fps_tutorial.py
python knn_tutorial.py
```
---

## API 문서

### torch_fpsample.sample()

> **sample** (x: torch.Tensor, num_samples: int, h: int = -1, start_idx: int = -1) 
    -> Tuple[torch.Tensor, torch.Tensor]**Parameters:**
- `x`: 입력 포인트 클라우드 `(batch_size, num_points, dimensions)`
- `num_samples`: 샘플링할 포인트 개수
- `h`: 트리 높이 (선택적, 기본값: -1은 자동)
- `start_idx`: 시작 포인트 인덱스 (선택적, 기본값: -1은 랜덤)

**Returns:**
- `sampled_points`: 샘플링된 포인트 `(batch_size, num_samples, dimensions)`
- `indices`: 선택된 포인트의 인덱스 `(batch_size, num_samples)`

### knn_torch3d.knn_points()

> **knn_points** (p1: torch.Tensor, p2: torch.Tensor, K: int = 1, 
           return_nn: bool = False, norm: int = 2) -> _KNN**Parameters:**
- `p1`: Query 포인트 `(N, P1, D)`
- `p2`: Reference 포인트 `(N, P2, D)`
- `K`: 찾을 최근접 이웃 개수
- `return_nn`: 이웃 좌표 반환 여부
- `norm`: 거리 노름 (1: L1, 2: L2)

**Returns:**
- `_KNN` NamedTuple with:
  - `dists`: 거리 `(N, P1, K)`
  - `idx`: 인덱스 `(N, P1, K)`
  - `knn`: 이웃 좌표 `(N, P1, K, D)` (return_nn=True일 때)

### knn_torch3d.knn_gather()

> **knn_gather** (x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor**Parameters:**
- `x`: 입력 텐서 `(N, M, U)`
- `idx`: 인덱스 텐서 `(N, L, K)`

**Returns:**
- `x_out`: 수집된 값 `(N, L, K, U)`

---

## 성능

> [!WARNING]
> 현재 GPU 버전은 구현되지 않았습니다. CPU 모드만 사용 가능합니다.

---

## 요구사항

- Python >= 3.8
- PyTorch >= 2.0
- C++ 컴파일러 (GCC, MSVC 등)
- OpenMP 지원 (선택적, 성능 향상)

---

## Reference

### FPS (Farthest Point Sampling)

Bucket-based farthest point sampling (QuickFPS)은 다음 논문에서 제안되었습니다:
btex
@article{han2023quickfps,
  title={QuickFPS: Architecture and Algorithm Co-Design for Farthest Point Sampling in Large-Scale Point Clouds},
  author={Han, Meng and Wang, Liang and Xiao, Limin and Zhang, Hao and Zhang, Chenhao and Xu, Xiangrong and Zhu, Jianfeng},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  year={2023},
  publisher={IEEE}
}

구현은 다음 저장소를 기반으로 합니다:
- [fpsample](https://github.com/leonardodalinky/pytorch_fpsample)

### KNN (K-Nearest Neighbors)

KNN 구현은 PyTorch3D의 KNN 알고리즘을 참고하였습니다.

---

## Author
**Hyunbin Cho**  
Email: hyunbin.cho@medit.com