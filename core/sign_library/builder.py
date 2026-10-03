"""Build script and algorithms to generate a sign library from landmark data.

Supports:
1. Medoid sequence selection per class: chooses the actual representative take
   that minimizes total distance to all other takes (preserves human biomechanics
   and realistic coarticulation without blur).
2. DTW-averaged sequence: time-warped averaging across samples.
3. Resampling to target keyframe length (default 48 frames).
4. Export to standard JSON format for the avatar player.
"""
import os
import sys
import glob
import json
import argparse
from typing import Dict, List, Optional, Union, Tuple
import numpy as np


def resample_sequence(seq: np.ndarray, target_len: int = 48) -> np.ndarray:
    """Linearly resample a (T, D) array to (target_len, D)."""
    t_orig = len(seq)
    if t_orig == target_len:
        return seq.copy().astype(np.float32)
    if t_orig == 0:
        return np.zeros((target_len, 225), dtype=np.float32)

    indices = np.linspace(0, t_orig - 1, target_len)
    idx_floor = np.floor(indices).astype(int)
    idx_ceil = np.minimum(idx_floor + 1, t_orig - 1)
    weight = (indices - idx_floor)[:, None]

    resampled = (1.0 - weight) * seq[idx_floor] + weight * seq[idx_ceil]
    return resampled.astype(np.float32)


def sequence_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """Euclidean distance between two (T, D) sequences."""
    return float(np.mean(np.linalg.norm(s1 - s2, axis=1)))


def dtw_distance(s1: np.ndarray, s2: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
    """Dynamic Time Warping (DTW) distance and alignment path between two sequences."""
    n, m = len(s1), len(s2)
    cost = np.zeros((n, m), dtype=np.float32)

    # Frame-to-frame distance matrix
    dist_matrix = np.linalg.norm(s1[:, None, :] - s2[None, :, :], axis=2)

    cost[0, 0] = dist_matrix[0, 0]
    for i in range(1, n):
        cost[i, 0] = cost[i - 1, 0] + dist_matrix[i, 0]
    for j in range(1, m):
        cost[0, j] = cost[0, j - 1] + dist_matrix[0, j]

    for i in range(1, n):
        for j in range(1, m):
            cost[i, j] = dist_matrix[i, j] + min(
                cost[i - 1, j],     # insertion
                cost[i, j - 1],     # deletion
                cost[i - 1, j - 1]  # match
            )

    # Backtrack alignment path
    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            steps = [(cost[i - 1, j - 1], i - 1, j - 1),
                     (cost[i - 1, j], i - 1, j),
                     (cost[i, j - 1], i, j - 1)]
            steps.sort(key=lambda x: x[0])
            _, i, j = steps[0]
        path.append((i, j))
    path.reverse()

    norm_cost = float(cost[n - 1, m - 1] / len(path))
    return norm_cost, path


def medoid_sequence(sequences: List[np.ndarray], target_len: int = 48) -> np.ndarray:
    """Finds the medoid sequence from an ensemble of landmark sequences.

    The medoid is the sequence that minimizes the sum of distances to all
    other sequences in the cluster.
    """
    if not sequences:
        raise ValueError("Cannot compute medoid of empty sequence list")

    norm_seqs = [resample_sequence(np.asarray(s, dtype=np.float32), target_len) for s in sequences]
    n = len(norm_seqs)
    if n == 1:
        return norm_seqs[0]

    dist_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = sequence_distance(norm_seqs[i], norm_seqs[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    total_dists = dist_matrix.sum(axis=1)
    best_idx = int(np.argmin(total_dists))
    return norm_seqs[best_idx]


def dtw_average_sequence(sequences: List[np.ndarray], target_len: int = 48) -> np.ndarray:
    """Computes a DTW-averaged sequence across an ensemble of takes."""
    if not sequences:
        raise ValueError("Cannot compute DTW average of empty sequence list")

    norm_seqs = [resample_sequence(np.asarray(s, dtype=np.float32), target_len) for s in sequences]
    if len(norm_seqs) == 1:
        return norm_seqs[0]

    # Use medoid as reference template
    reference = medoid_sequence(sequences, target_len)
    aligned_accum = np.zeros((target_len, 225), dtype=np.float32)
    aligned_counts = np.zeros((target_len, 1), dtype=np.float32)

    for s in norm_seqs:
        _, path = dtw_distance(reference, s)
        for ref_i, s_j in path:
            aligned_accum[ref_i] += s[s_j]
            aligned_counts[ref_i] += 1.0

    aligned_counts = np.maximum(aligned_counts, 1.0)
    avg_seq = aligned_accum / aligned_counts
    return avg_seq.astype(np.float32)


def build_library_from_dict(
    data: Dict[str, List[np.ndarray]],
    method: str = "medoid",
    target_len: int = 48
) -> Dict[str, List[List[float]]]:
    """Generates a library dict {sign_id: [[float, ... 225], ... 48 frames]} from raw sequences."""
    library: Dict[str, List[List[float]]] = {}
    for sign_id, seq_list in data.items():
        if not seq_list:
            continue
        if method == "dtw":
            rep = dtw_average_sequence(seq_list, target_len)
        else:
            rep = medoid_sequence(seq_list, target_len)

        # Convert to standard Python float list
        library[sign_id] = np.round(rep, 4).tolist()

    return library


def build_library_from_dir(
    input_dir: str,
    output_path: str,
    method: str = "medoid",
    target_len: int = 48
) -> Dict[str, List[List[float]]]:
    """Scans landmark data directory (e.g., data/processed/<class>/*.npy) and builds library JSON."""
    data: Dict[str, List[np.ndarray]] = {}

    if os.path.exists(input_dir):
        # Look for subdirectories or class-named npy files
        for root, dirs, files in os.walk(input_dir):
            npy_files = [f for f in files if f.endswith(".npy")]
            if not npy_files:
                continue
            class_name = os.path.basename(root)
            if class_name not in data:
                data[class_name] = []
            for f in npy_files:
                path = os.path.join(root, f)
                try:
                    arr = np.load(path)
                    data[class_name].append(arr)
                except Exception as ex:
                    print(f"Warning: could not load {path}: {ex}", file=sys.stderr)

    # Fallback to existing signs.json if directory has no npy files
    if not data:
        models_signs = os.path.join(os.path.dirname(__file__), "..", "..", "models", "signs.json")
        if os.path.exists(models_signs):
            print(f"Using existing library baseline from {models_signs}")
            with open(models_signs, "r") as f:
                raw = json.load(f)
            data = {k: [np.array(v, dtype=np.float32)] for k, v in raw.items()}

    library = build_library_from_dict(data, method=method, target_len=target_len)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(library, f)

    print(f"Successfully wrote sign library with {len(library)} signs to {output_path}")
    return library


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build sign keyframe library from landmark data")
    parser.add_argument("--input", default="data/processed", help="Input directory containing landmark .npy takes")
    parser.add_argument("--output", default="models/signs.json", help="Output path for library JSON")
    parser.add_argument("--method", choices=["medoid", "dtw"], default="medoid", help="Aggregation algorithm")
    args = parser.parse_args()

    build_library_from_dir(args.input, args.output, method=args.method)
