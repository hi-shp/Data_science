import numpy as np
from scipy.ndimage import label
from config import GRID, GRID_W, GRID_H

# 고속 연산을 위한 그리드 인덱스 배열 사전 생성
_Y_INDICES, _X_INDICES = np.indices((GRID_H, GRID_W), dtype=np.float32)
_Y_FLAT = _Y_INDICES.ravel()
_X_FLAT = _X_INDICES.ravel()

def lidar_hits_np(boat_pos, boat_heading, rel_angles, obstacles, lidar_range):
    if len(obstacles) == 0:
        n = len(rel_angles)
        return np.full(n, lidar_range, np.float32), [None] * n

    ox = obstacles[:, 0:1].T
    oy = obstacles[:, 1:2].T
    orad = obstacles[:, 2:3].T

    angs = (boat_heading + rel_angles)[:, None]
    vx = np.cos(angs)
    vy = np.sin(angs)

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
    hits = [(float(hx), float(hy)) if v else None for hx, hy, v in zip(hits_x, hits_y, valid)]

    return d_final, hits

def init_grid():
    return np.zeros((GRID_H, GRID_W), dtype=np.float32)

def update_grid(grid, hits):
    valid_pts = [p for p in hits if p is not None]
    if not valid_pts:
        return
    pts = np.asarray(valid_pts, dtype=np.float32)
    gx = (pts[:, 0] // GRID).astype(np.int32)
    gy = (pts[:, 1] // GRID).astype(np.int32)
    valid = (gx >= 0) & (gx < GRID_W) & (gy >= 0) & (gy < GRID_H)
    if np.any(valid):
        np.add.at(grid, (gy[valid], gx[valid]), 1.0)
        np.clip(grid, 0.0, 20.0, out=grid)

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