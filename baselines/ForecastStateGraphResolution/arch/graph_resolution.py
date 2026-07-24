"""Distance-aware affinity and capacity-constrained graph resolution.

Paper: Eq. (10)–(18). Caches assignments under datasets/<name>/forecast_state_graph_cache/.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

IMPLEMENTATION_VERSION = "fsgr_graph_v1"


def num_clusters_for_capacity(num_nodes: int, capacity: int) -> int:
    cap = max(1, int(capacity))
    return (int(num_nodes) + cap - 1) // cap


def load_raw_adjacency(adj_mx_path: str) -> np.ndarray:
    with open(adj_mx_path, "rb") as f:
        try:
            obj = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            obj = pickle.load(f, encoding="latin1")
    if isinstance(obj, (list, tuple)):
        for item in reversed(obj):
            if hasattr(item, "shape") and len(getattr(item, "shape")) == 2:
                obj = item
                break
    adj = np.asarray(obj, dtype=np.float64)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Invalid adjacency shape: {adj.shape}")
    return adj


def symmetrize_adjacency(adj: np.ndarray) -> np.ndarray:
    return 0.5 * (adj + adj.T)


def adjacency_hash(adj: np.ndarray) -> str:
    arr = np.ascontiguousarray(adj, dtype=np.float64)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]


def shortest_path_distance(adj_sym: np.ndarray) -> np.ndarray:
    """Unweighted shortest-path distance on support of adj_sym (>0)."""
    n = adj_sym.shape[0]
    connected = (adj_sym > 0).astype(np.float64)
    np.fill_diagonal(connected, 0.0)
    # BFS from each node
    dist = np.full((n, n), np.inf, dtype=np.float64)
    for i in range(n):
        dist[i, i] = 0.0
        queue = [i]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            nbrs = np.where(connected[u] > 0)[0]
            for v in nbrs:
                if dist[i, v] > dist[i, u] + 1:
                    dist[i, v] = dist[i, u] + 1
                    queue.append(v)
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return np.zeros_like(dist)
    # unreachable -> large finite
    max_d = float(finite.max()) if finite.size else 1.0
    dist[~np.isfinite(dist)] = max_d + 1.0
    return dist


def normalize_distances(dist: np.ndarray) -> np.ndarray:
    flat = dist[np.triu_indices_from(dist, k=1)]
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return np.zeros_like(dist)
    q95 = float(np.quantile(flat, 0.95))
    q95 = max(q95, 1e-6)
    return np.clip(dist / q95, 0.0, 1.0)


def try_load_physical_distance(
    num_nodes: int,
    distance_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    if not distance_path or not os.path.exists(distance_path):
        return None
    path = Path(distance_path)
    if path.suffix == ".csv":
        # sensor_id,sensor_id,distance triples or square matrix
        raw = np.loadtxt(path, delimiter=",", dtype=np.float64)
        if raw.ndim == 2 and raw.shape[0] == num_nodes and raw.shape[1] == num_nodes:
            return raw
        if raw.ndim == 2 and raw.shape[1] >= 3:
            dist = np.full((num_nodes, num_nodes), np.inf)
            np.fill_diagonal(dist, 0.0)
            for row in raw:
                i, j, d = int(row[0]), int(row[1]), float(row[2])
                if 0 <= i < num_nodes and 0 <= j < num_nodes:
                    dist[i, j] = min(dist[i, j], d)
                    dist[j, i] = min(dist[j, i], d)
            return dist
        return None
    with open(path, "rb") as f:
        obj = pickle.load(f)
    arr = np.asarray(obj, dtype=np.float64)
    if arr.shape == (num_nodes, num_nodes):
        return arr
    return None


def build_distance_aware_affinity(
    adj: np.ndarray,
    sigma_d: float = 0.5,
    distance_path: Optional[str] = None,
    distance_mode: str = "auto",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """W_ij = A_sym_ij * exp(-d_norm^2 / sigma_d^2) with fallback priority."""
    a_sym = symmetrize_adjacency(adj)
    n = a_sym.shape[0]
    meta: dict[str, Any] = {"distance_mode_requested": distance_mode}
    dist = None
    mode_used = None

    if distance_mode in {"auto", "physical"}:
        dist = try_load_physical_distance(n, distance_path)
        if dist is not None:
            mode_used = "physical"

    if dist is None and distance_mode in {"auto", "adjacency_as_similarity", "shortest_path"}:
        # PEMS adj_mx is similarity in [0,1]; use shortest-path on support
        if distance_mode == "adjacency_as_similarity" or distance_mode == "auto":
            dist = shortest_path_distance(a_sym)
            mode_used = "shortest_path"
        elif distance_mode == "shortest_path":
            dist = shortest_path_distance(a_sym)
            mode_used = "shortest_path"

    if dist is None:
        warnings.warn(
            "[FSGR] No usable distance matrix; falling back to adjacency-only affinity W=A_sym.",
            RuntimeWarning,
        )
        w = a_sym.copy()
        np.fill_diagonal(w, 0.0)
        meta.update(
            {
                "distance_mode_used": "adjacency_only",
                "warning": "adjacency_only_fallback",
            }
        )
        return w.astype(np.float64), np.zeros_like(a_sym), meta

    d_norm = normalize_distances(dist)
    sigma = max(float(sigma_d), 1e-6)
    w = a_sym * np.exp(-(d_norm ** 2) / (sigma ** 2))
    np.fill_diagonal(w, 0.0)
    meta.update(
        {
            "distance_mode_used": mode_used,
            "sigma_d": sigma,
            "distance_path": distance_path or "",
        }
    )
    return w.astype(np.float64), d_norm.astype(np.float64), meta


def _spectral_embedding(w: np.ndarray, dim: int, seed: int = 0) -> np.ndarray:
    n = w.shape[0]
    dim = max(1, min(int(dim), n - 1))
    w_bar = w + np.eye(n)
    deg = w_bar.sum(axis=1)
    deg_inv_sqrt = np.power(np.maximum(deg, 1e-12), -0.5)
    d_mat = np.diag(deg_inv_sqrt)
    s = d_mat @ w_bar @ d_mat
    # eigh for symmetric
    vals, vecs = np.linalg.eigh(s)
    idx = np.argsort(vals)[::-1]
    vecs = vecs[:, idx[:dim]]
    # row normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    rng = np.random.RandomState(seed)
    # slight jitter for determinism of ties downstream
    return (vecs / norms) + 0.0 * rng.randn(*vecs.shape)


def _kmeans_centers(features: np.ndarray, k: int, seed: int, n_init: int = 5) -> np.ndarray:
    rng = np.random.RandomState(seed)
    n, d = features.shape
    best_centers = None
    best_inertia = np.inf
    for init in range(n_init):
        centers = features[rng.choice(n, size=k, replace=False)].copy()
        labels = np.zeros(n, dtype=np.int64)
        for _ in range(30):
            dist = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = dist.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                mask = labels == j
                if mask.any():
                    centers[j] = features[mask].mean(axis=0)
                else:
                    centers[j] = features[rng.randint(0, n)]
        inertia = ((features - centers[labels]) ** 2).sum()
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.copy()
    return best_centers


def _capacity_greedy_assign(
    features: np.ndarray,
    centers: np.ndarray,
    capacity: int,
    dist_norm: Optional[np.ndarray],
    lambda_d: float,
    seed: int,
) -> np.ndarray:
    """Assign each node to nearest center with free capacity (deterministic)."""
    n, m = features.shape[0], centers.shape[0]
    emb_cost = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    if dist_norm is not None and float(lambda_d) > 0:
        # approximate medoid = nearest center index's closest node later; use center proxy via argmin emb
        # use average distance to provisional members unavailable; use distance to seed medoids = argmin to center
        medoids = emb_cost.argmin(axis=0)  # for each cluster, closest node as medoid proxy
        emb_cost = emb_cost + float(lambda_d) * dist_norm[:, medoids]

    remaining = np.full(m, int(capacity), dtype=np.int64)
    # Ensure total capacity >= n
    if remaining.sum() < n:
        raise RuntimeError(
            f"Total capacity {remaining.sum()} < num_nodes {n}; increase clusters or capacity."
        )
    labels = np.full(n, -1, dtype=np.int64)
    # order nodes by best-minus-second-best gap (harder first)
    order_scores = np.partition(emb_cost, 1, axis=1)[:, :2]
    urgency = order_scores[:, 1] - order_scores[:, 0]
    order = np.argsort(-urgency)
    rng = np.random.RandomState(seed)
    # stable tie-break
    for node in order:
        pref = np.argsort(emb_cost[node], kind="mergesort")
        assigned = False
        for cl in pref:
            if remaining[cl] > 0:
                labels[node] = int(cl)
                remaining[cl] -= 1
                assigned = True
                break
        if not assigned:
            raise RuntimeError("Capacity-constrained assignment failed.")
    # repair empty clusters by stealing from largest
    for cl in range(m):
        if (labels == cl).any():
            continue
        sizes = np.bincount(labels, minlength=m)
        donor = int(np.argmax(sizes))
        donors = np.where(labels == donor)[0]
        # steal farthest from donor center
        dcost = ((features[donors] - centers[donor]) ** 2).sum(axis=1)
        steal = int(donors[int(np.argmax(dcost))])
        labels[steal] = cl
    return labels


def _local_swap_refine(
    features: np.ndarray,
    labels: np.ndarray,
    capacity: int,
    max_passes: int = 3,
) -> np.ndarray:
    labels = labels.copy()
    n, m = features.shape[0], int(labels.max()) + 1
    centers = np.stack([features[labels == k].mean(0) if (labels == k).any() else features.mean(0) for k in range(m)])
    for _ in range(max_passes):
        improved = False
        for i in range(n):
            cur = labels[i]
            cur_cost = ((features[i] - centers[cur]) ** 2).sum()
            sizes = np.bincount(labels, minlength=m)
            best_cl, best_gain = cur, 0.0
            for cl in range(m):
                if cl == cur:
                    continue
                if sizes[cl] >= capacity:
                    continue
                new_cost = ((features[i] - centers[cl]) ** 2).sum()
                gain = cur_cost - new_cost
                if gain > best_gain + 1e-12:
                    best_gain, best_cl = gain, cl
            if best_cl != cur:
                labels[i] = best_cl
                improved = True
                # update centers lightly
                for k in (cur, best_cl):
                    mask = labels == k
                    if mask.any():
                        centers[k] = features[mask].mean(0)
        if not improved:
            break
    return labels


def labels_to_assignment(labels: np.ndarray, num_nodes: int, num_clusters: int) -> np.ndarray:
    c = np.zeros((num_nodes, num_clusters), dtype=np.float64)
    c[np.arange(num_nodes), labels] = 1.0
    return c


def assignment_to_projection(c: np.ndarray) -> np.ndarray:
    sizes = c.sum(axis=0)
    if np.any(sizes <= 0):
        raise RuntimeError(f"Empty cluster detected in assignment: sizes={sizes}")
    d_inv = 1.0 / sizes
    return (c * d_inv[None, :]).T  # [M, N]


def validate_assignment(
    labels: np.ndarray,
    capacity: int,
    adj_sym: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    n = len(labels)
    m = int(labels.max()) + 1 if n > 0 else 0
    sizes = np.bincount(labels, minlength=m)
    over = int((sizes > capacity).sum())
    empty = int((sizes == 0).sum())
    disconnected = 0
    if adj_sym is not None:
        for k in range(m):
            members = np.where(labels == k)[0]
            if members.size <= 1:
                continue
            sub = adj_sym[np.ix_(members, members)] > 0
            # BFS connectivity ignoring weights
            seen = {0}
            stack = [0]
            while stack:
                u = stack.pop()
                for v in range(len(members)):
                    if v not in seen and sub[u, v]:
                        seen.add(v)
                        stack.append(v)
            if len(seen) != len(members):
                disconnected += 1
    checksum = hashlib.sha1(labels.astype(np.int64).tobytes()).hexdigest()[:16]
    report = {
        "num_clusters": m,
        "min_cluster_size": int(sizes.min()) if sizes.size else 0,
        "mean_cluster_size": float(sizes.mean()) if sizes.size else 0.0,
        "max_cluster_size": int(sizes.max()) if sizes.size else 0,
        "over_capacity_count": over,
        "empty_cluster_count": empty,
        "disconnected_cluster_count": disconnected,
        "assignment_checksum": checksum,
        "capacity": int(capacity),
    }
    if over > 0 or empty > 0:
        raise RuntimeError(f"Hard capacity/assignment constraint violated: {report}")
    return report


def identity_assignment(num_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = np.eye(num_nodes, dtype=np.float64)
    p = np.eye(num_nodes, dtype=np.float64)
    labels = np.arange(num_nodes, dtype=np.int64)
    return c, p, labels


def build_capacity_constrained_clusters(
    adj: np.ndarray,
    capacity: int,
    seed: int = 1,
    sigma_d: float = 0.5,
    lambda_d: float = 0.1,
    distance_path: Optional[str] = None,
    distance_mode: str = "auto",
) -> dict[str, Any]:
    n = adj.shape[0]
    cap = int(capacity)
    if cap <= 1:
        c, p, labels = identity_assignment(n)
        w, d_norm, aff_meta = build_distance_aware_affinity(
            adj, sigma_d=sigma_d, distance_path=distance_path, distance_mode=distance_mode
        )
        report = validate_assignment(labels, capacity=1, adj_sym=symmetrize_adjacency(adj))
        return {
            "C": c.astype(np.float32),
            "P": p.astype(np.float32),
            "labels": labels,
            "W": w.astype(np.float32),
            "num_clusters": n,
            "capacity": 1,
            "validation": report,
            "affinity_meta": aff_meta,
        }

    w, d_norm, aff_meta = build_distance_aware_affinity(
        adj, sigma_d=sigma_d, distance_path=distance_path, distance_mode=distance_mode
    )
    m = num_clusters_for_capacity(n, cap)
    z = _spectral_embedding(w, dim=min(m, n - 1), seed=seed)
    centers = _kmeans_centers(z, m, seed=seed)
    labels = _capacity_greedy_assign(z, centers, cap, d_norm, lambda_d, seed=seed)
    labels = _local_swap_refine(z, labels, cap)
    # recompute centers and one more assignment pass
    centers = np.stack([z[labels == k].mean(0) for k in range(m)])
    labels = _capacity_greedy_assign(z, centers, cap, d_norm, lambda_d, seed=seed + 7)
    labels = _local_swap_refine(z, labels, cap)
    a_sym = symmetrize_adjacency(adj)
    report = validate_assignment(labels, capacity=cap, adj_sym=a_sym)
    c = labels_to_assignment(labels, n, m)
    p = assignment_to_projection(c)
    return {
        "C": c.astype(np.float32),
        "P": p.astype(np.float32),
        "labels": labels.astype(np.int64),
        "W": w.astype(np.float32),
        "num_clusters": m,
        "capacity": cap,
        "validation": report,
        "affinity_meta": aff_meta,
    }


def cache_key(
    dataset: str,
    adj_hash: str,
    capacity: int,
    distance_mode: str,
    sigma_d: float,
    seed: int,
) -> str:
    payload = {
        "dataset": dataset,
        "adjacency_hash": adj_hash,
        "capacity": int(capacity),
        "distance_mode": distance_mode,
        "sigma_d": float(sigma_d),
        "seed": int(seed),
        "implementation_version": IMPLEMENTATION_VERSION,
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def load_or_build_graph_resolution(
    dataset: str,
    adj_mx_path: str,
    capacity: int,
    seed: int = 1,
    sigma_d: float = 0.5,
    lambda_d: float = 0.1,
    distance_path: Optional[str] = None,
    distance_mode: str = "auto",
    cache_dir: Optional[str] = None,
) -> dict[str, Any]:
    adj = load_raw_adjacency(adj_mx_path)
    a_hash = adjacency_hash(adj)
    cache_root = Path(
        cache_dir
        or os.path.join("datasets", dataset, "forecast_state_graph_cache")
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    key = cache_key(dataset, a_hash, capacity, distance_mode, sigma_d, seed)
    cache_path = cache_root / f"{dataset}_cap{capacity}_seed{seed}_{key}.npz"
    if cache_path.is_file():
        data = np.load(cache_path, allow_pickle=True)
        meta = {
            "C": data["C"].astype(np.float32),
            "P": data["P"].astype(np.float32),
            "labels": data["labels"].astype(np.int64),
            "W": data["W"].astype(np.float32),
            "num_clusters": int(data["num_clusters"]),
            "capacity": int(data["capacity"]),
            "validation": data["validation"].item() if "validation" in data else {},
            "affinity_meta": data["affinity_meta"].item() if "affinity_meta" in data else {},
            "cache_path": str(cache_path),
            "from_cache": True,
        }
        # re-validate hard constraints
        validate_assignment(meta["labels"], capacity=meta["capacity"])
        return meta

    meta = build_capacity_constrained_clusters(
        adj,
        capacity=capacity,
        seed=seed,
        sigma_d=sigma_d,
        lambda_d=lambda_d,
        distance_path=distance_path,
        distance_mode=distance_mode,
    )
    np.savez_compressed(
        cache_path,
        C=meta["C"],
        P=meta["P"],
        labels=meta["labels"],
        W=meta["W"],
        num_clusters=meta["num_clusters"],
        capacity=meta["capacity"],
        validation=meta["validation"],
        affinity_meta=meta["affinity_meta"],
        adjacency_hash=a_hash,
        implementation_version=IMPLEMENTATION_VERSION,
    )
    meta["cache_path"] = str(cache_path)
    meta["from_cache"] = False
    print(
        f"[FSGR] Built graph resolution capacity={capacity} "
        f"M={meta['num_clusters']} validation={meta['validation']} "
        f"affinity={meta['affinity_meta']}"
    )
    return meta


def build_structural_graph(p: np.ndarray, w: np.ndarray, c: np.ndarray) -> np.ndarray:
    """A_str = RowNorm(P W C + I)."""
    a = p @ w @ c
    a = a + np.eye(a.shape[0], dtype=np.float64)
    row = a.sum(axis=1, keepdims=True)
    row = np.maximum(row, 1e-8)
    return (a / row).astype(np.float32)
