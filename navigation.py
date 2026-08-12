import numpy as np
import math
from utils import wrap
from config import GRID, GRID_W, GRID_H

def front_is_clear(boat_pos, boat_heading, obstacles, check_dist=300, fov=np.deg2rad(25)):
    bx, by = boat_pos
    for (ox, oy, r) in obstacles:
        dx = ox - bx
        dy = oy - by
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > check_dist:
            continue
        ang = math.atan2(dy, dx)
        rel = wrap(ang - boat_heading)
        if abs(rel) < fov:
            return False
    return True

def find_gap(clusters, ids, boat_pos, boat_heading, gps_heading, visited, grid, obstacles):
    if front_is_clear(boat_pos, boat_heading, obstacles):
        return None
        
    bx, by = boat_pos
    gps_vec = np.array([math.cos(gps_heading), math.sin(gps_heading)])
    
    items = []
    for i, c in enumerate(clusters):
        v = c - boat_pos
        dist = np.linalg.norm(v)
        ang = wrap(math.atan2(v[1], v[0]) - boat_heading)
        if abs(ang) < np.pi/2:
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
        path_clear = min(min_clear / 120, 1)**2.5
        
        cnt = 0
        for (ox2, oy2, r2) in obs_f:
            if (ox2 - mx)**2 + (oy2 - my)**2 < 100*100:
                cnt += 1
        cluster_pen = math.exp(-0.5 * cnt)
        
        gap_w = np.linalg.norm(c2 - c1)
        width_w = min(gap_w / 90, 1)
        small_gap = math.exp(-gap_w / 40)
        
        sc = heading_align**4.5 * forward_proj**1.5 * lateral_full**2 * path_clear**2.5 * width_w**1.2 * cluster_pen * small_gap
        
        if sc > best_sc:
            best_sc = sc
            best = {"pos": mid, "c1": c1.copy(), "c2": c2.copy(), "pair": (id1, id2), "score": sc}
            
    return best

def reactive_avoidance(dists, angles):
    SAFE = 360
    sigma = 120
    a = 0.
    for d, ang in zip(dists, angles):
        if d < SAFE:
            w = math.exp(-(d / sigma)**2)
            front = max(1.1 - abs(ang) / (math.pi/2), 0.4)
            a -= w * front * math.sin(ang)
    return a