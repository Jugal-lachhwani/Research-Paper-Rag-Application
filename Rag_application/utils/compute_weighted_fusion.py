def compute_weighted_fusion(dense_list, sparse_list, alpha=0.4):
    """
    Computes Weighted Fusion across two lists of Qdrant points.
    Alpha controls the weight of the dense_list, and (1-alpha) controls sparse_list.
    Returns a single sorted list of points.
    """
    dense_scores = {p.id: p.score for p in dense_list}
    sparse_scores = {p.id: p.score for p in sparse_list}
    
    # Normalize scores (Min-Max scaling)
    def normalize(scores):
        if not scores:
            return scores
        min_val = min(scores.values())
        max_val = max(scores.values())
        if max_val == min_val:
            return {k: 1.0 for k in scores}
        return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}
    norm_dense = normalize(dense_scores)
    norm_sparse = normalize(sparse_scores)
    
    fused_scores = {}
    point_map = {}
    
    for p in dense_list:
        point_map[p.id] = p
        fused_scores[p.id] = alpha * norm_dense.get(p.id, 0)
        
    for p in sparse_list:
        point_map[p.id] = p
        if p.id in fused_scores:
            fused_scores[p.id] += (1 - alpha) * norm_sparse.get(p.id, 0)
        else:
            fused_scores[p.id] = (1 - alpha) * norm_sparse.get(p.id, 0)
            
    # Sort by fused score descending
    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    return [point_map[pid] for pid in sorted_ids]
     