import math
import numpy as np
from scipy.ndimage import label
from config import GRID, GRID_W, GRID_H

# 고속 연산을 위한 그리드 인덱스 배열 및 라이다 각도 테이블 사전 생성
_Y_INDICES, _X_INDICES = np.indices((GRID_H, GRID_W), dtype=np.float32)
_Y_FLAT = _Y_INDICES.ravel()
_X_FLAT = _X_INDICES.ravel()

_REL_ANGLES = np.linspace(-np.pi, np.pi, 180, endpoint=False, dtype=np.float32)
_COS_REL = np.cos(_REL_ANGLES)[:, None]
_SIN_REL = np.sin(_REL_ANGLES)[:, None]

def lidar_hits_np(boat_pos, boat_heading, rel_angles, obstacles, lidar_range):
    if len(obstacles) == 0:
        n = len(rel_angles)
        return np.full(n, lidar_range, np.float32), [None] * n

    ox = obstacles[:, 0:1].T
    oy = obstacles[:, 1:2].T
    orad = obstacles[:, 2:3].T

    ch = math.cos(boat_heading)
    sh = math.sin(boat_heading)
    vx = ch * _COS_REL - sh * _SIN_REL
    vy = sh * _COS_REL + ch * _SIN_REL

    x0, y0 = boat_pos
    px = ox - x0
    py = oy - y0

    b = px * vx + py * vy
    perp2 = (px - b * vx)**2 + (py - b * vy)**2
    disc = orad**2 - perp2

    mask = (b > 0) & (disc >= 0)
    t = np.where(mask, b - np.sqrt(np.maximum(0, disc)), lidar_range)
    t = np.where(t > 0, t, lidar_range)

    d_final = np.min(t, axis=1).astype(np.float32)
    
    valid = d_final < lidar_range
    hits_x = x0 + vx[:, 0] * d_final
    hits_y = y0 + vy[:, 0] * d_final
    hits = [None] * len(d_final)
    for idx in np.where(valid)[0]:
        hits[idx] = (float(hits_x[idx]), float(hits_y[idx]))

    return d_final, hits

def init_grid():
    return np.zeros((GRID_H, GRID_W), dtype=np.float32)

def update_grid(grid, hits):
    for p in hits:
        if p is not None:
            gx = int(p[0] // GRID)
            gy = int(p[1] // GRID)
            if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                if grid[gy, gx] < 20.0:
                    grid[gy, gx] += 1.0

def extract_clusters_from_grid(grid):
    OCC = 2.0
    mask = grid >= OCC
    if not np.any(mask):
        return []
    labeled_array, num_features = label(mask)
    if num_features == 0:
        return []
    
    flat_labels = labeled_array.ravel()
    counts = np.bincount(flat_labels, minlength=num_features + 1)[1:]
    sum_y = np.bincount(flat_labels, weights=_Y_FLAT, minlength=num_features + 1)[1:]
    sum_x = np.bincount(flat_labels, weights=_X_FLAT, minlength=num_features + 1)[1:]
    
    valid = counts > 0
    cy = sum_y[valid] / counts[valid]
    cx = sum_x[valid] / counts[valid]
    
    world_x = cx * GRID + GRID / 2.0
    world_y = cy * GRID + GRID / 2.0
    return [np.array([wx, wy], dtype=np.float32) for wx, wy in zip(world_x, world_y)]

def match_clusters(prev_clusters, prev_ids, new_clusters):
    if len(prev_clusters) == 0:
        return new_clusters, list(range(len(new_clusters)))
    
    cell_prev = {}
    for cid, c in zip(prev_ids, prev_clusters):
        key = (int(c[0] // GRID), int(c[1] // GRID))
        cell_prev[key] = cid
        
    new_ids = []
    maxid = max(prev_ids) + 1 if prev_ids else 0
    
    for c in new_clusters:
        key = (int(c[0] // GRID), int(c[1] // GRID))
        if key in cell_prev:
            new_ids.append(cell_prev[key])
        else:
            new_ids.append(maxid)
            maxid += 1
            
    return new_clusters, new_ids