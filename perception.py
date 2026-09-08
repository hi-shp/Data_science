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

def lidar_hits_np(boat_pos, boat_heading, rel_angles, obstacles, lidar_range, map_bounds=None):
    n = len(rel_angles)
    ch = math.cos(boat_heading)
    sh = math.sin(boat_heading)
    vx = ch * _COS_REL - sh * _SIN_REL
    vy = sh * _COS_REL + ch * _SIN_REL

    x0, y0 = boat_pos

    if len(obstacles) > 0:
        ox = obstacles[:, 0:1].T
        oy = obstacles[:, 1:2].T
        orad = obstacles[:, 2:3].T

        px = ox - x0
        py = oy - y0

        b = px * vx + py * vy
        perp2 = (px - b * vx)**2 + (py - b * vy)**2
        disc = orad**2 - perp2

        mask = (b > 0) & (disc >= 0)
        t = np.where(mask, b - np.sqrt(np.maximum(0, disc)), lidar_range)
        t = np.where(t > 0, t, lidar_range)

        d_final = np.min(t, axis=1).astype(np.float32)
    else:
        d_final = np.full(n, lidar_range, dtype=np.float32)

    # 맵 외곽 벽(Boundary Walls)을 장애물로 인식
    if map_bounds is not None:
        xmin, ymin, xmax, ymax = map_bounds
        t_left = np.where(vx < -1e-5, (xmin - x0) / np.minimum(vx, -1e-5), lidar_range)
        t_right = np.where(vx > 1e-5, (xmax - x0) / np.maximum(vx, 1e-5), lidar_range)
        t_top = np.where(vy < -1e-5, (ymin - y0) / np.minimum(vy, -1e-5), lidar_range)
        t_bottom = np.where(vy > 1e-5, (ymax - y0) / np.maximum(vy, 1e-5), lidar_range)

        t_left = np.where(t_left > 0, t_left, lidar_range)
        t_right = np.where(t_right > 0, t_right, lidar_range)
        t_top = np.where(t_top > 0, t_top, lidar_range)
        t_bottom = np.where(t_bottom > 0, t_bottom, lidar_range)

        t_wall = np.minimum(np.minimum(t_left, t_right), np.minimum(t_top, t_bottom))
        d_final = np.minimum(d_final, t_wall[:, 0].astype(np.float32))
    
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
    coords = [p for p in hits if p is not None]
    if not coords:
        return
    arr = np.array(coords, dtype=np.float32)
    gx = (arr[:, 0] / GRID).astype(np.intp)
    gy = (arr[:, 1] / GRID).astype(np.intp)
    valid = (gx >= 0) & (gx < GRID_W) & (gy >= 0) & (gy < GRID_H)
    gx = gx[valid]
    gy = gy[valid]
    for k in range(len(gx)):
        if grid[gy[k], gx[k]] < 20.0:
            grid[gy[k], gx[k]] += 1.0

from sklearn.cluster import DBSCAN

def extract_clusters_from_grid(grid):
    OCC = 1.0
    gy, gx = np.where(grid >= OCC)
    if len(gx) == 0:
        return []
    
    world_x = gx * GRID + GRID / 2.0
    world_y = gy * GRID + GRID / 2.0
    pts = np.column_stack((world_x, world_y))
    weights = grid[gy, gx]
    
    # DBSCAN 클러스터링: 장애물 반경 17px(지름 34px) 내에 찍힌 모든 라이다 히트점을 하나의 덩어리로 병합
    db = DBSCAN(eps=42.0, min_samples=1).fit(pts)
    labels = db.labels_
    
    clusters = []
    for lbl in sorted(set(labels)):
        mask = (labels == lbl)
        w = weights[mask]
        total_w = float(np.sum(w))
        if total_w > 0:
            cx = float(np.sum(world_x[mask] * w) / total_w)
            cy = float(np.sum(world_y[mask] * w) / total_w)
            clusters.append(np.array([cx, cy], dtype=np.float32))
            
    return clusters

def match_clusters(prev_clusters, prev_ids, new_clusters, max_dist=28.0):
    if len(new_clusters) == 0:
        return [], []
    if len(prev_clusters) == 0 or len(prev_ids) == 0:
        return new_clusters, list(range(len(new_clusters)))
    
    n_new = len(new_clusters)
    n_prev = len(prev_clusters)
    
    # 거리 행렬을 벡터 연산으로 일괄 계산
    new_arr = np.array(new_clusters).reshape(n_new, 2)
    prev_arr = np.array(prev_clusters).reshape(n_prev, 2)
    diff = new_arr[:, None, :] - prev_arr[None, :, :]
    dist_mat = np.sqrt(np.sum(diff * diff, axis=2))
    
    new_ids = [0] * n_new
    used_prev = set()
    maxid = max(prev_ids) + 1 if prev_ids else 0
    
    for i in range(n_new):
        row = dist_mat[i]
        best_idx = None
        best_d = max_dist
        for j in range(n_prev):
            if j in used_prev:
                continue
            if row[j] < best_d:
                best_d = row[j]
                best_idx = j
        if best_idx is not None:
            new_ids[i] = prev_ids[best_idx]
            used_prev.add(best_idx)
        else:
            new_ids[i] = maxid
            maxid += 1
            
    return new_clusters, new_ids