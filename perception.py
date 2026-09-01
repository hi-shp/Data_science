import numpy as np
from scipy.ndimage import label
from config import GRID, GRID_W, GRID_H

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
    
    hits = []
    for i, d in enumerate(d_final):
        if d < lidar_range:
            hits.append((float(x0 + vx[i, 0] * d), float(y0 + vy[i, 0] * d)))
        else:
            hits.append(None)

    return d_final, hits

def init_grid():
    return np.zeros((GRID_H, GRID_W), dtype=np.float32)

def update_grid(grid, hits):
    for p in hits:
        if p is None: continue
        gx = int(p[0] // GRID)
        gy = int(p[1] // GRID)
        if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
            grid[gy, gx] = min(grid[gy, gx] + 1.0, 20.0)

def extract_clusters_from_grid(grid):
    OCC = 2.0
    mask = grid >= OCC
    if not np.any(mask):
        return []
    labeled_array, num_features = label(mask)
    clusters = []
    for lb in range(1, num_features + 1):
        ys, xs = np.where(labeled_array == lb)
        if len(xs) < 2:
            continue
        cx = np.mean(xs) * GRID + GRID / 2.0
        cy = np.mean(ys) * GRID + GRID / 2.0
        clusters.append(np.array([cx, cy], dtype=np.float32))
    return clusters

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