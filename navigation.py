import numpy as np
import math
from utils import wrap
from config import GRID, GRID_W, GRID_H

def target_is_clear(boat_pos, target_pos, obstacles, boat_radius=25):
    bx, by = boat_pos
    tx, ty = target_pos
    vx = tx - bx
    vy = ty - by
    dist_t2 = vx * vx + vy * vy
    
    if dist_t2 < 1e-6:
        return True
        
    ox = obstacles[:, 0]
    oy = obstacles[:, 1]
    # 직선 판정 안전 여유를 10px로 확대하여 장애물 인근 아슬아슬한 직진 방지
    orad = obstacles[:, 2] + boat_radius + 10 
    
    px = ox - bx
    py = oy - by
    t = (px * vx + py * vy) / dist_t2
    
    valid = (t >= 0.0) & (t <= 1.0)
    if not np.any(valid):
        return True
        
    cx = bx + t[valid] * vx
    cy = by + t[valid] * vy
    d2 = (ox[valid] - cx)**2 + (oy[valid] - cy)**2
    
    return not np.any(d2 <= orad[valid]**2)

def find_gap(clusters, ids, boat_pos, boat_heading, target_pos, visited, grid, obstacles):
    bx, by = boat_pos
    tx, ty = target_pos
    dx_t = tx - bx
    dy_t = ty - by
    dist_to_target = math.hypot(dx_t, dy_t)
    gps_heading = math.atan2(dy_t, dx_t)

    if dist_to_target < 150 or target_is_clear(boat_pos, target_pos, obstacles):
        return None
        
    gps_vec = np.array([math.cos(gps_heading), math.sin(gps_heading)])
    
    items = []
    for i, c in enumerate(clusters):
        v = c - boat_pos
        dist = np.linalg.norm(v)
        # 클러스터가 목적지보다 멀리 있으면 무시 (엄격하게 차단)
        if dist > dist_to_target + 15:
            continue
            
        ang = wrap(math.atan2(v[1], v[0]) - boat_heading)
        if abs(ang) < np.pi * 0.8:
            items.append((ang, dist, c, ids[i]))
            
    if len(items) < 2:
        return None
        
    items.sort(key=lambda x: x[0])
    
    gaps = []
    for i in range(len(items) - 1):
        if (items[i+1][0] - items[i][0]) > np.deg2rad(2.0):
            gaps.append((i, i+1))
            
    if not gaps:
        return None
        
    best = None
    best_sc = -1
    ox = obstacles[:, 0]
    oy = obstacles[:, 1]
    
    for gi, gj in gaps:
        ang1, d1, c1, id1 = items[gi]
        ang2, d2, c2, id2 = items[gj]
        
        if (id1, id2) in visited or (id2, id1) in visited:
            continue
            
        mid = (c1 + c2) / 2
        mx, my = mid
        rel = mid - boat_pos
        distm = np.linalg.norm(rel) + 1e-6
        
        # 웨이포인트(mid)가 목적지보다 멀어지면 무조건 제외하여 엉뚱한 우회 방지
        if distm > dist_to_target + 15:
            continue
            
        gx = int(mx // GRID)
        gy = int(my // GRID)
        blocked = False
        for dy_grid in range(-2, 3):
            for dx_grid in range(-2, 3):
                yy = gy + dy_grid
                xx = gx + dx_grid
                if 0 <= xx < GRID_W and 0 <= yy < GRID_H:
                    if grid[yy, xx] >= 3.0:
                        blocked = True
                        break
            if blocked: break
        if blocked: continue
        
        ang_mid = math.atan2(rel[1], rel[0])
        ang_err = wrap(ang_mid - gps_heading)
        
        heading_align = math.exp(-(ang_err / 0.9)**2)
        
        forward_proj = np.dot(rel / distm, gps_vec)
        forward_proj = max(forward_proj, 0)**1.5
        
        lateral = abs(ang2 - ang1) / (np.pi/2)
        lateral = min(max(lateral, 0), 1)**2
        
        sym = 1 - abs(abs(ang1) - abs(ang2)) / (np.pi/2)
        sym = min(max(sym, 0), 1)
        
        lateral_full = 0.6 * lateral + 0.4 * sym
        
        vx = mx - bx
        vy = my - by
        seg2 = distm * distm
        d2_obs = (ox - bx)**2 + (oy - by)**2
        
        mask = d2_obs <= (distm + 200)**2
        obs_f = obstacles[mask]
        
        min_clear = 9999
        for (ox2, oy2, r2) in obs_f:
            px = ox2 - bx
            py = oy2 - by
            t = (px*vx + py*vy) / seg2
            t = max(0, min(1, t))
            cx = bx + t*vx
            cy = by + t*vy
            d = math.sqrt((ox2 - cx)**2 + (oy2 - cy)**2) - r2
            if d < min_clear:
                min_clear = d
                
        min_clear = max(min_clear, 0)
        path_clear = min(min_clear / 150, 1)**2.5 
        
        cnt = 0
        for (ox2, oy2, r2) in obs_f:
            if (ox2 - mx)**2 + (oy2 - my)**2 < 100*100:
                cnt += 1
        cluster_pen = math.exp(-0.5 * cnt)
        
        dir_x = mx - bx
        dir_y = my - by
        depth_pen = 1.0
        if distm > 10:
            norm_x = dir_x / distm
            norm_y = dir_y / distm
            past_x = mx + norm_x * 120
            past_y = my + norm_y * 120
            
            past_blocked = 0
            for (ox2, oy2, r2) in obs_f:
                if (ox2 - past_x)**2 + (oy2 - past_y)**2 < 80*80:
                    past_blocked += 1
            
            depth_pen = math.exp(-1.5 * past_blocked)
        
        gap_w = np.linalg.norm(c2 - c1)
        width_w = min(gap_w / 90, 1)
        
        sc = heading_align**5 * forward_proj**3 * lateral_full**0.5 * path_clear**1.5 * width_w**0.2 * cluster_pen * depth_pen
        
        if sc > best_sc:
            best_sc = sc
            best = {"pos": mid, "c1": c1.copy(), "c2": c2.copy(), "pair": (id1, id2), "score": sc}
            
    return best

def reactive_avoidance(dists, angles):
    SAFE = 450
    sigma = 200 
    a = 0.
    for d, ang in zip(dists, angles):
        if d < SAFE:
            w = math.exp(-(d / sigma)**2)
            front = max(1.2 - abs(ang) / (math.pi/2), 0.3)
            a -= w * front * math.sin(ang)
    return a