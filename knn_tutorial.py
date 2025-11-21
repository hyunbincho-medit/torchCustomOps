import torch
from knn_torch3d import knn_points, knn_gather
from validation import verify_knn_results


# 샘플 텐서 생성
batch_size = 2
N1 = 100  # p1의 포인트 수
N2 = 200  # p2의 포인트 수
D = 3     # 차원 (x, y, z)

p1 = torch.randn(batch_size, N1, D)  # query points
p2 = torch.randn(batch_size, N2, D)  # reference points

# KNN 계산
result = knn_points(p1, p2, K=5, return_nn=True)
dists = result.dists  # 거리
idx = result.idx      # 인덱스
knn = result.knn      # nearest neighbors (return_nn=True일 때)

# 또는 knn_gather 사용
neighbors = knn_gather(p2, idx)

# TorchScript로 변환 및 export
class KNNModule(torch.nn.Module):
    def __init__(self, K=5):
        super().__init__()
        self.K = K
    
    def forward(self, p1, p2):
        result = knn_points(p1, p2, K=self.K, return_nn=False)
        return result.dists, result.idx

class KNNGatherModule(torch.nn.Module):
    def __init__(self, K=5):
        super().__init__()
        self.K = K
    
    def forward(self, p1, p2):
        result = knn_points(p1, p2, K=self.K, return_nn=False)
        neighbors = knn_gather(p2, result.idx)
        return result.dists, result.idx, neighbors

# 모델 생성 및 스크립팅
knn_module = KNNModule(K=5)
scripted_module = torch.jit.script(knn_module)

# knn_gather 포함 모델
knn_gather_module = KNNGatherModule(K=5)
scripted_gather_module = torch.jit.script(knn_gather_module)

# 모델 저장
scripted_module.save("knn_model.pt")
scripted_gather_module.save("knn_gather_model.pt")
print("TorchScript 모델이 'knn_model.pt'와 'knn_gather_model.pt'로 저장되었습니다.")

# 저장된 모델 로드 및 테스트
loaded_model = torch.jit.load("knn_model.pt")
test_dists, test_idx = loaded_model(p1, p2)
print(f"Distance shape: {test_dists.shape}, Index shape: {test_idx.shape}")

# knn_gather 모델 테스트
loaded_gather_model = torch.jit.load("knn_gather_model.pt")
test_dists, test_idx, test_neighbors = loaded_gather_model(p1, p2)
print(f"Neighbors shape: {test_neighbors.shape}")

result = knn_points(p1, p2, K=5, return_nn=True)
dists = result.dists
idx = result.idx
knn = result.knn

# knn_gather 사용
neighbors = knn_gather(p2, idx)

# 검증 함수 호출
verification_results = verify_knn_results(
    p1=p1,
    p2=p2,
    dists=dists,
    idx=idx,
    neighbors=neighbors,
    knn=knn,
    K=5,
    num_samples=5,
    verbose=True
)

# 검증 결과 확인
if verification_results['all_passed']:
    print("\n✅ KNN 연산이 올바르게 작동합니다!")
else:
    print("\n❌ KNN 연산에 문제가 있습니다.")
    print(f"상세 결과: {verification_results}")