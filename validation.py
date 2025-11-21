import torch

def verify_knn_results(p1, p2, dists, idx, neighbors, knn=None, K=5, num_samples=5, verbose=True):
    """
    KNN 연산 결과를 검증하는 함수
    
    Args:
        p1: Query points (batch_size, N1, D)
        p2: Reference points (batch_size, N2, D)
        dists: KNN 거리 결과 (batch_size, N1, K)
        idx: KNN 인덱스 결과 (batch_size, N1, K)
        neighbors: knn_gather 결과 (batch_size, N1, K, D)
        knn: knn_points에서 return_nn=True일 때의 결과 (선택적)
        K: 이웃 개수
        num_samples: 검증할 샘플 포인트 개수 (검증 1에서 사용)
        verbose: 상세 출력 여부
    
    Returns:
        dict: 각 검증 항목의 결과를 담은 딕셔너리
    """
    batch_size, N1, D = p1.shape
    N2 = p2.shape[1]
    
    results = {
        'distance_accuracy': True,
        'sorting_correct': True,
        'gather_correct': True,
        'knn_match': True,
        'boundary_check': True,
        'index_valid': True,
        'all_passed': True
    }
    
    if verbose:
        print("=" * 60)
        print("KNN 연산 검증 시작")
        print("=" * 60)
    
    # 1. knn_points의 dists와 idx 검증
    if verbose:
        print("\n[검증 1] knn_points 거리 계산 정확도 검증")
    
    for b in range(batch_size):
        for i in range(min(num_samples, N1)):
            query_point = p1[b, i]
            manual_dists = torch.sum((p2[b] - query_point) ** 2, dim=-1)
            knn_dists = dists[b, i]
            knn_indices = idx[b, i]
            extracted_dists = manual_dists[knn_indices]
            
            is_close = torch.allclose(knn_dists, extracted_dists, rtol=1e-5, atol=1e-5)
            if not is_close:
                results['distance_accuracy'] = False
                if verbose:
                    print(f"⚠️  배치 {b}, 포인트 {i}: 거리 불일치 발견!")
                    print(f"   KNN 거리: {knn_dists}")
                    print(f"   직접 계산: {extracted_dists}")
    
    if verbose and results['distance_accuracy']:
        print("✓ knn_points 거리 검증 완료")
    
    # 2. 거리가 오름차순으로 정렬되어 있는지 검증
    if verbose:
        print("\n[검증 2] 거리 정렬 확인")
    
    is_sorted = torch.all(dists[:, :, :-1] <= dists[:, :, 1:])
    results['sorting_correct'] = is_sorted.item()
    
    if verbose:
        if results['sorting_correct']:
            print("✓ 모든 거리가 오름차순으로 정렬되어 있습니다.")
        else:
            print("⚠️  거리가 올바르게 정렬되지 않았습니다!")
    
    # 3. knn_gather 검증
    if verbose:
        print("\n[검증 3] knn_gather 정확도 검증")
    
    manual_neighbors = torch.zeros_like(neighbors)
    for b in range(batch_size):
        for i in range(N1):
            for k in range(K):
                neighbor_idx = idx[b, i, k]
                manual_neighbors[b, i, k] = p2[b, neighbor_idx]
    
    is_gather_correct = torch.allclose(neighbors, manual_neighbors, rtol=1e-5, atol=1e-5)
    results['gather_correct'] = is_gather_correct  # .item() 제거
    
    if verbose:
        if results['gather_correct']:
            print("✓ knn_gather가 올바르게 작동합니다.")
        else:
            print("⚠️  knn_gather 결과가 예상과 다릅니다!")
            max_diff = torch.max(torch.abs(neighbors - manual_neighbors))
            print(f"   최대 차이: {max_diff}")
    
    # 4. return_nn=True일 때 knn과 knn_gather 결과 비교
    if verbose:
        print("\n[검증 4] knn_points의 knn과 knn_gather 결과 비교")
    
    if knn is not None:
        is_knn_equal = torch.allclose(knn, neighbors, rtol=1e-5, atol=1e-5)
        results['knn_match'] = is_knn_equal  # .item() 제거
        
        if verbose:
            if results['knn_match']:
                print("✓ knn_points의 knn과 knn_gather 결과가 일치합니다.")
            else:
                print("⚠️  결과가 일치하지 않습니다!")
                max_diff = torch.max(torch.abs(knn - neighbors))
                print(f"   최대 차이: {max_diff}")
    else:
        results['knn_match'] = None
        if verbose:
            print("ℹ️  knn이 None입니다. (return_nn=True로 설정하면 검증 가능)")
    
    # 5. 경계 케이스 검증
    if verbose:
        print("\n[검증 5] 경계 케이스 검증")
    
    min_dist = torch.min(dists)
    results['min_distance'] = min_dist.item()
    results['boundary_check'] = min_dist >= 1e-6
    
    if verbose:
        print(f"   최소 거리: {min_dist:.6f}")
        if not results['boundary_check']:
            print("⚠️  거의 동일한 위치의 포인트가 발견되었습니다.")
        else:
            print("✓ 모든 거리가 유효합니다.")
    
    # 6. 인덱스 범위 검증
    if verbose:
        print("\n[검증 6] 인덱스 범위 검증")
    
    is_valid_range = torch.all((idx >= 0) & (idx < N2))
    results['index_valid'] = is_valid_range.item()  # 이것도 유지
    
    if verbose:
        if results['index_valid']:
            print(f"✓ 모든 인덱스가 유효한 범위 [0, {N2}) 내에 있습니다.")
        else:
            print(f"⚠️  범위를 벗어난 인덱스가 있습니다!")
            print(f"   최소 인덱스: {torch.min(idx)}, 최대 인덱스: {torch.max(idx)}")
    
    # 7. 통계 정보
    results['stats'] = {
        'mean_nearest_dist': dists[:, :, 0].mean().item(),
        'mean_kth_dist': dists[:, :, -1].mean().item(),
        'dist_std': dists.std().item()
    }
    
    if verbose:
        print("\n[통계 정보]")
        print(f"   평균 최근접 이웃 거리: {results['stats']['mean_nearest_dist']:.6f}")
        print(f"   평균 K번째 이웃 거리: {results['stats']['mean_kth_dist']:.6f}")
        print(f"   거리 표준편차: {results['stats']['dist_std']:.6f}")
    
    # 전체 결과 판정
    results['all_passed'] = (
        results['distance_accuracy'] and
        results['sorting_correct'] and
        results['gather_correct'] and
        (results['knn_match'] if results['knn_match'] is not None else True) and
        results['index_valid']
    )
    
    if verbose:
        print("\n" + "=" * 60)
        if results['all_passed']:
            print("✓✓✓ 모든 검증 통과! ✓✓✓")
        else:
            print("⚠️  일부 검증 실패")
        print("=" * 60)
    
    return results