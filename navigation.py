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
        
    dist_t = math.sqrt(dist_t2)
    ux = vx / dist_t
    uy = vy / dist_t
    
    ox = obstacles[:, 0]
    oy = obstacles[:, 1]
    orad = obstacles[:, 2] + boat_radius + 4.0  # 안전 여유 4px
    
    px = ox - bx
    py = oy - by
    proj = px * ux + py * uy
    
    # 선박 바로 옆에 있거나(proj < 10) 이미 지나친 장애물은 전방 직선 경로를 가로막지 않음
    valid = (proj >= 10.0) & (proj <= dist_t)
    if not np.any(valid):
        return True
        
    cx = bx + proj[valid] * ux
    cy = by + proj[valid] * uy
    d2 = (ox[valid] - cx)**2 + (oy[valid] - cy)**2
    
    return not np.any(d2 <= orad[valid]**2)

def bezier_path_is_blocked(path, obstacles, boat_radius=25, margin=10):
    if path is None or len(path) == 0 or len(obstacles) == 0:
        return False
    ox = obstacles[:, 0]
    oy = obstacles[:, 1]
    orad2 = (obstacles[:, 2] + boat_radius + margin) ** 2
    for p in path:
        px, py = p[0], p[1]
        if np.any((ox - px)**2 + (oy - py)**2 <= orad2):
            return True
    return False

def is_direct_target_safe(boat_pos, boat_heading, target_pos, obstacles, boat_radius=25, boat_speed=0.0):
    if not target_is_clear(boat_pos, target_pos, obstacles, boat_radius=boat_radius):
        return False
    from utils import make_bezier_path
    # 목적지까지 회전하는 실제 베지어 궤적이 장애물과 충돌하는지 검증
    test_path = make_bezier_path(boat_pos, boat_heading, target_pos, obstacles=obstacles, boat_radius=boat_radius, boat_speed=boat_speed)
    if bezier_path_is_blocked(test_path, obstacles, boat_radius=boat_radius, margin=8.0):
        return False
    return True

def is_waypoint_switch_safe(boat_pos, boat_heading, curr_wp_pos, new_wp_pos, obstacles, boat_radius=25, boat_speed=0.0):
    if curr_wp_pos is None or new_wp_pos is None or len(obstacles) == 0:
        return True
        
    bx, by = boat_pos
    v_curr = curr_wp_pos - boat_pos
    v_new = new_wp_pos - boat_pos
    
    ang_curr = math.atan2(v_curr[1], v_curr[0])
    ang_new = math.atan2(v_new[1], v_new[0])
    
    # 각도 차이가 작으면 (동일 방향/미세 갱신) 안전
    switch_ang_diff = abs(wrap(ang_new - ang_curr))
    if switch_ang_diff < np.deg2rad(25.0):
        return True
        
    # 1. 현재 선박 헤딩에서 새 웨이포인트로 선회하는 베지어 곡선 검증
    from utils import make_bezier_path
    new_bezier = make_bezier_path(boat_pos, boat_heading, new_wp_pos, obstacles=obstacles, boat_radius=boat_radius, boat_speed=boat_speed)
    if bezier_path_is_blocked(new_bezier, obstacles, boat_radius=boat_radius, margin=8.0):
        return False
        
    # 2. 기존 방향과 새 방향 사이의 부채꼴(Turn Sector) 영역 장애물 검사
    ang_head_to_new = wrap(ang_new - boat_heading)
    sweep_radius = 110.0
    
    for ox, oy, orad in obstacles:
        dx = ox - bx
        dy = oy - by
        obs_dist = math.hypot(dx, dy)
        
        if obs_dist - orad < sweep_radius:
            ang_obs = math.atan2(dy, dx)
            rel_ang = wrap(ang_obs - boat_heading)
            
            in_sector = False
            if ang_head_to_new >= 0:
                if -0.15 <= rel_ang <= ang_head_to_new + 0.15:
                    in_sector = True
            else:
                if ang_head_to_new - 0.15 <= rel_ang <= 0.15:
                    in_sector = True
                    
            if in_sector:
                if obs_dist - orad < boat_radius + 40.0:
                    return False
                    
    return True

def is_front_blocked(boat_pos, boat_heading, obstacles, boat_radius=25, block_dist=130.0, fov_deg=110.0):
    if obstacles is None or len(obstacles) == 0:
        return False
    bx, by = boat_pos
    fov_rad = np.deg2rad(fov_deg)  # 좌우 45도 = 총 90도 전방 부채꼴
    
    for ox, oy, orad in obstacles:
        dx = ox - bx
        dy = oy - by
        dist = math.hypot(dx, dy)
        clear_dist = dist - orad
        if clear_dist < block_dist:
            ang_to_obs = math.atan2(dy, dx)
            rel_ang = abs(wrap(ang_to_obs - boat_heading))
            if rel_ang <= fov_rad:
                return True
    return False

def find_gap(clusters, ids, boat_pos, boat_heading, target_pos, visited, grid, obstacles):
    bx, by = boat_pos
    tx, ty = target_pos
    dx_t = tx - bx
    dy_t = ty - by
    dist_to_target = math.hypot(dx_t, dy_t)
    gps_heading = math.atan2(dy_t, dx_t)

    gps_vec = np.array([math.cos(gps_heading), math.sin(gps_heading)])
    
    items = []
    for i, c in enumerate(clusters):
        v = c - boat_pos
        dist = np.linalg.norm(v)
        # 목적지보다 멀거나 전방 탐색각(65도)을 벗어난 측방/후방 장애물 엄격 제외
        if dist > dist_to_target - 20:
            continue
            
        ang = wrap(math.atan2(v[1], v[0]) - boat_heading)
        if abs(ang) < np.deg2rad(65):
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
        
    valid_gaps = []
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
        dist_mid_to_target = np.linalg.norm(target_pos - mid)
        
        # 1. 갭(mid)이 현재 탐색 기준 위치보다 목적지에 유의미하게 가까워져야 함
        if dist_mid_to_target >= dist_to_target - 15.0:
            continue
            
        forward_progress = np.dot(rel / distm, gps_vec)
        # 목적지 방향 전진 성분이 부족하거나(측면/후방 회피) 목적지보다 멀면 제외
        min_progress = 0.55 if dist_to_target < 300 else 0.40
        if forward_progress < min_progress or distm > dist_to_target - 25:
            continue
            
        ang_mid = math.atan2(rel[1], rel[0])
        ang_err = wrap(ang_mid - gps_heading)
        
        # 목적지 방향과 40도 이상 어긋나는 무리한 측방 갭 제외
        if abs(ang_err) > np.deg2rad(40):
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
        
        if len(obs_f) > 0:
            px = obs_f[:, 0] - bx
            py = obs_f[:, 1] - by
            t = np.clip((px * vx + py * vy) / seg2, 0.0, 1.0)
            cx = bx + t * vx
            cy = by + t * vy
            dists_to_seg = np.sqrt((obs_f[:, 0] - cx)**2 + (obs_f[:, 1] - cy)**2) - obs_f[:, 2]
            min_clear = float(np.min(dists_to_seg))
            
            # 보트 눈앞(160px 이내) 장애물과의 간격이 좁을 경우(45px 미만) 선제적 페널티 부여
            d_boat = np.hypot(px, py)
            close_mask = (d_boat < 160.0) & (dists_to_seg < 45.0)
            if np.any(close_mask):
                close_d = np.maximum(dists_to_seg[close_mask], 0.0)
                near_clear_penalty = float(np.prod(np.maximum(0.1, (close_d / 45.0) ** 1.5)))
            else:
                near_clear_penalty = 1.0
                
            cnt = int(np.sum((obs_f[:, 0] - mx)**2 + (obs_f[:, 1] - my)**2 < 10000.0))
            cluster_pen = math.exp(-0.5 * cnt)
            
            depth_pen = 1.0
            if distm > 10:
                dir_x = mx - bx
                dir_y = my - by
                norm_x = dir_x / distm
                norm_y = dir_y / distm
                past_x = mx + norm_x * 120
                past_y = my + norm_y * 120
                past_blocked = int(np.sum((obs_f[:, 0] - past_x)**2 + (obs_f[:, 1] - past_y)**2 < 6400.0))
                depth_pen = math.exp(-1.5 * past_blocked)
        else:
            min_clear = 9999.0
            near_clear_penalty = 1.0
            cluster_pen = 1.0
            depth_pen = 1.0
                
        min_clear = max(min_clear, 0)
        path_clear = min(min_clear / 160, 1)**2.2
        
        gap_w = np.linalg.norm(c2 - c1)
        width_w = min(gap_w / 90, 1)
        
        sc = (heading_align**4.5) * (forward_proj**2.8) * (lateral_full**0.5) * (path_clear**2.0) * (width_w**0.2) * cluster_pen * depth_pen * near_clear_penalty
        
        if sc > 0:
            valid_gaps.append({
                "pos": mid,
                "c1": c1.copy(),
                "c2": c2.copy(),
                "pair": (id1, id2),
                "score": sc
            })
            
    if not valid_gaps:
        return None
        
    valid_gaps.sort(key=lambda x: x["score"], reverse=True)
    best = valid_gaps[0]
    best["candidates"] = valid_gaps[1:3]  # 순위 높은 2, 3위 차순위 후보
    return best

def reactive_avoidance(dists, angles):
    SAFE = 450.0
    sigma = 150.0
    mask = dists < SAFE
    if not np.any(mask):
        return 0.0
    d = dists[mask]
    ang = angles[mask]
    w = np.exp(-((d / sigma)**2))
    front = np.maximum(1.2 - np.abs(ang) / (np.pi / 2.0), 0.3)
    return float(np.sum(-w * front * np.sin(ang)))