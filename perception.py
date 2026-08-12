import numpy as np
from sklearn.cluster import DBSCAN
from config import GRID, GRID_W, GRID_H

def lidar_hits_np(boat_pos, boat_heading, rel_angles, obstacles, lidar_range):
    if len(obstacles) == 0:
        n = len(rel_angles)
        d = np.full(n, lidar_range, np.float32)
        return d, [None]*n
    ox = obstacles[:, 0]
    oy = obstacles[:, 1]
    orad = obstacles[:, 2]
    angs = boat_heading + rel_angles
    dx = np.cos(angs)
    dy = np.sin(angs)
    d_final = np.full_like(angs, lidar_range, dtype=np.float32)
    hits = [None] * len(angs)
    
    x0 = boat_pos[0]
    y0 = boat_pos[1]

    for i in range(len(angs)):
        vx = dx[i]
        vy = dy[i]
        px = ox - x0
        py = oy - y0
        b = px*vx + py*vy
        mask = b > 0
        if not np.any(mask):
            continue
        
        b2 = b[mask]
        px2 = px[mask]
        py2 = py[mask]
        r2 = orad[mask]
        
        perp = px2 - b2*vx
        perp2 = py2 - b2*vy
        dist2 = perp*perp + perp2*perp2
        hitmask = dist2 <= r2*r2
        
        if not np.any(hitmask):
            continue
            
        b3 = b2[hitmask]
        dist2_2 = dist2[hitmask]
        r3 = r2[hitmask]
        
        f = np.sqrt(np.maximum(0, r3*r3 - dist2_2))
        t = b3 - f
        
        valid_t = t[(t > 0) & (t < d_final[i])]
        
        if len(valid_t) > 0:
            tmin = np.min(valid_t)
            d_final[i] = tmin
            hx = x0 + vx * tmin
            hy = y0 + vy * tmin
            hits[i] = (hx, hy)
            
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
    ys, xs = np.where(grid >= OCC)
    if len(xs) == 0:
        return []
    pts = np.column_stack([(xs * GRID + GRID/2), (ys * GRID + GRID/2)]).astype(np.float32)
    if len(pts) == 0:
        return []
    
    db = DBSCAN(eps=22, min_samples=2).fit(pts)
    labels = db.labels_
    clusters = []
    for lb in set(labels):
        if lb == -1:
            continue
        mask = (labels == lb)
        clusters.append(np.mean(pts[mask], axis=0))
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