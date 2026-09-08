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
    orad = obstacles[:, 2] + boat_radius - 4.0
    
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
    # 벡터화: 모든 경로 점과 장애물 사이 거리를 한 번에 계산
    px = path[:, 0][:, None]  # (N, 1)
    py = path[:, 1][:, None]
    d2 = (ox[None, :] - px)**2 + (oy[None, :] - py)**2  # (N, M)
    return bool(np.any(d2 <= orad2[None, :]))

def is_direct_target_safe(boat_pos, boat_heading, target_pos, obstacles, boat_radius=25, boat_speed=0.0, params=None):
    # 1. 목적지 방향 직선 경로(시야) 확보 여부 1차 검사
    if not target_is_clear(boat_pos, target_pos, obstacles, boat_radius=boat_radius):
        return False
    if obstacles is None or len(obstacles) == 0:
        return True
        
    # 2. 현재 헤딩에서 목적지로 꺾을 때 선회 궤적(베지어 곡선)에 장애물이 없는지 검증
    from utils import make_bezier_path
    p = params or {}
    margin = float(p.get('clear_margin', 10.0))
    test_path = make_bezier_path(boat_pos, boat_heading, target_pos, obstacles=obstacles, boat_radius=boat_radius, boat_speed=boat_speed)
    if bezier_path_is_blocked(test_path, obstacles, boat_radius=boat_radius, margin=margin):
        return False
    return True

def is_waypoint_switch_safe(boat_pos, boat_heading, curr_wp_pos, new_wp_pos, obstacles, boat_radius=25, boat_speed=0.0, params=None):
    if curr_wp_pos is None or new_wp_pos is None or len(obstacles) == 0:
        return True
        
    bx, by = boat_pos
    v_curr = curr_wp_pos - boat_pos
    v_new = new_wp_pos - boat_pos
    
    ang_curr = math.atan2(v_curr[1], v_curr[0])
    ang_new = math.atan2(v_new[1], v_new[0])
    
    # 각도 차이가 작으면 (동일 방향/미세 갱신) 안전
    switch_ang_diff = abs(wrap(ang_new - ang_curr))
    if switch_ang_diff < 0.4363323129985824:  # deg2rad(25.0)
        return True
        
    # 1. 현재 선박 헤딩에서 새 웨이포인트로 선회하는 베지어 곡선 검증
    from utils import make_bezier_path
    p = params or {}
    margin = float(p.get('clear_margin', 10.0))
    new_bezier = make_bezier_path(boat_pos, boat_heading, new_wp_pos, obstacles=obstacles, boat_radius=boat_radius, boat_speed=boat_speed)
    if bezier_path_is_blocked(new_bezier, obstacles, boat_radius=boat_radius, margin=margin):
        return False
        
    # 2. 기존 방향과 새 방향 사이의 부채꼴(Turn Sector) 영역 장애물 검사
    ang_head_to_new = wrap(ang_new - boat_heading)
    sweep_radius = 110.0
    
    dx_all = obstacles[:, 0] - bx
    dy_all = obstacles[:, 1] - by
    dist_all = np.sqrt(dx_all * dx_all + dy_all * dy_all)
    close_mask = (dist_all - obstacles[:, 2]) < sweep_radius
    if np.any(close_mask):
        for k in np.where(close_mask)[0]:
            dx = dx_all[k]
            dy = dy_all[k]
            orad = obstacles[k, 2]
            obs_dist = dist_all[k]
            
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

def is_front_blocked(boat_pos, boat_heading, obstacles, boat_radius=25, block_dist=190.0, fov_deg=130.0):
    if obstacles is None or len(obstacles) == 0:
        return False
    bx, by = boat_pos
    fov_rad = np.deg2rad(fov_deg)
    
    dx = obstacles[:, 0] - bx
    dy = obstacles[:, 1] - by
    dist = np.sqrt(dx * dx + dy * dy)
    clear_dist = dist - obstacles[:, 2]
    close_mask = clear_dist < block_dist
    if not np.any(close_mask):
        return False
    ang_to_obs = np.arctan2(dy[close_mask], dx[close_mask])
    rel_ang = np.abs((ang_to_obs - boat_heading + np.pi) % (2 * np.pi) - np.pi)
    return bool(np.any(rel_ang <= fov_rad))

def find_gap(clusters, ids, boat_pos, boat_heading, target_pos, visited, grid, obstacles, params=None, is_next_wp=False):
    bx, by = boat_pos
    tx, ty = target_pos
    dx_t = tx - bx
    dy_t = ty - by
    dist_to_target = math.hypot(dx_t, dy_t)
    gps_heading = math.atan2(dy_t, dx_t)

    align_exp = params.get('align_exp', 6.0) if params else 6.0
    fwd_exp = params.get('fwd_exp', 6.6) if params else 6.6
    clear_exp = params.get('clear_exp', 5.0) if params else 5.0
    width_exp = params.get('width_exp', 0.2) if params else 0.2
    heading_exp = params.get('heading_exp', params.get('boat_align_exp', params.get('head_exp', 2.0))) if params else 2.0
    perp_exp = params.get('perp_exp', 3.0) if params else 3.0
    prox_exp = params.get('prox_exp', 4.0) if params else 4.0
    center_exp = params.get('center_exp', 1.5) if params else 1.5

    gps_vec = np.array([math.cos(gps_heading), math.sin(gps_heading)])
    
    h_cos = math.cos(boat_heading)
    h_sin = math.sin(boat_heading)
    h_vec = np.array([h_cos, h_sin], dtype=np.float32)
    
    max_ang = 1.4835298641951802 if is_next_wp else 1.1344640137963142  # deg2rad(85) / deg2rad(65)
    max_dist_cut = (dist_to_target + 15) if is_next_wp else (dist_to_target - 20)

    items = []
    for i, c in enumerate(clusters):
        v = c - boat_pos
        dist = math.hypot(v[0], v[1])
        # 목적지보다 멀거나 전방 탐색각을 벗어난 측방/후방 장애물 엄격 제외
        if dist > max_dist_cut:
            continue
            
        ang = wrap(math.atan2(v[1], v[0]) - boat_heading)
        if abs(ang) < max_ang:
            items.append((ang, dist, c, ids[i]))
            
    if len(items) < 2:
        return None
        
    items.sort(key=lambda x: x[0])
    
    ox = obstacles[:, 0]
    oy = obstacles[:, 1]
    
    gaps_set = set()
    # 1) 라이다 각도 상 분리된 인접 쌍 (라이다 상 검은색 빈 공간)
    for i in range(len(items) - 1):
        if (items[i+1][0] - items[i][0]) > 0.03490658503988659:  # deg2rad(2.0)
            gaps_set.add((i, i+1))
            
    # 2) 3개 이상의 장애물 조합(1-2, 2-3뿐만 아니라 1-3, 1-4 등) 및 깊이 단차가 있는 모든 가능한 틈새 조합 탐색
    # O(N)으로 각 클러스터와 장애물 간 거리 제곱을 사전 계산하여 O(N^2) 중복 연산 제거
    d2_items = [(ox - it[2][0])**2 + (oy - it[2][1])**2 for it in items]
    
    for i in range(len(items)):
        c1 = items[i][2]
        d2_c1 = d2_items[i]
        for j in range(i + 1, len(items)):
            c2 = items[j][2]
            v_gap = c2 - c1
            gap_w = math.hypot(v_gap[0], v_gap[1])
            
            # 최소 통과 폭 (45px) ~ 전방 게이트 유효 최대 폭 (280px)
            if not (45.0 <= gap_w <= 280.0):
                continue
                
            # 바운딩 박스 빠른 필터링: c1과 c2 영역 바깥에 있는 장애물은 검사 대상에서 즉시 배제
            min_x = (c1[0] if c1[0] < c2[0] else c2[0]) - 25.0
            max_x = (c1[0] if c1[0] > c2[0] else c2[0]) + 25.0
            min_y = (c1[1] if c1[1] < c2[1] else c2[1]) - 25.0
            max_y = (c1[1] if c1[1] > c2[1] else c2[1]) + 25.0
            
            mask_obs = (d2_c1 > 784.0) & (d2_items[j] > 784.0) & (ox >= min_x) & (ox <= max_x) & (oy >= min_y) & (oy <= max_y)
            if np.any(mask_obs):
                near_obs = obstacles[mask_obs]
                px = near_obs[:, 0] - c1[0]
                py = near_obs[:, 1] - c1[1]
                t = (px * v_gap[0] + py * v_gap[1]) / (gap_w * gap_w + 1e-6)
                in_span = (t > 0.05) & (t < 0.95)
                if np.any(in_span):
                    cand_obs = near_obs[in_span]
                    cand_t = t[in_span]
                    cx = c1[0] + cand_t * v_gap[0]
                    cy = c1[1] + cand_t * v_gap[1]
                    dist_to_gate = np.sqrt((cand_obs[:, 0] - cx)**2 + (cand_obs[:, 1] - cy)**2) - cand_obs[:, 2]
                    if np.any(dist_to_gate < 15.0):
                        # 게이트 사이가 제3의 장애물로 가로막혀 있으므로 단일 갭으로 취급하지 않음
                        continue
                        
            gaps_set.add((i, j))
                
    gaps = sorted(list(gaps_set))
    if not gaps:
        return None
        
    valid_gaps = []
    
    for gi, gj in gaps:
        ang1, d1, c1, id1 = items[gi]
        ang2, d2, c2, id2 = items[gj]
        
        if (id1, id2) in visited or (id2, id1) in visited:
            continue
            
        mid = (c1 + c2) / 2
        mx, my = mid
        rel = mid - boat_pos
        distm = math.hypot(rel[0], rel[1]) + 1e-6
        dist_mid_to_target = math.hypot(tx - mx, ty - my)
        
        # 1. 갭(mid)이 현재 탐색 기준 위치보다 목적지에 유의미하게 가까워져야 함
        req_progress_dist = -5.0 if is_next_wp else -15.0
        if dist_mid_to_target >= dist_to_target + req_progress_dist:
            continue
            
        forward_progress = np.dot(rel / distm, gps_vec)
        # 목적지 방향 전진 성분이 부족하거나(측면/후방 회피) 목적지보다 멀면 제외
        min_progress = 0.25 if is_next_wp else (0.55 if dist_to_target < 300 else 0.40)
        max_allowed_distm = (dist_to_target + 10) if is_next_wp else (dist_to_target - 25)
        if forward_progress < min_progress or distm > max_allowed_distm:
            continue
            
            
        ang_mid = math.atan2(rel[1], rel[0])
        ang_err = wrap(ang_mid - gps_heading)
        ang_boat_err = wrap(ang_mid - boat_heading)
        head_score = math.exp(-(ang_boat_err / 0.8)**2)
        head_factor = max(head_score, 0.05) ** heading_exp
        
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
            # c1, c2(게이트 기둥) 자체는 통과 대상이므로 보트->중점 선분 장애물 간섭 검사에서 제외
            d_to_c1 = (obs_f[:, 0] - c1[0])**2 + (obs_f[:, 1] - c1[1])**2
            d_to_c2 = (obs_f[:, 0] - c2[0])**2 + (obs_f[:, 1] - c2[1])**2
            other_mask = (d_to_c1 > 28.0**2) & (d_to_c2 > 28.0**2)
            obs_path = obs_f[other_mask]

            if len(obs_path) > 0:
                px = obs_path[:, 0] - bx
                py = obs_path[:, 1] - by
                t = np.clip((px * vx + py * vy) / seg2, 0.0, 1.0)
                cx = bx + t * vx
                cy = by + t * vy
                dists_to_seg = np.sqrt((obs_path[:, 0] - cx)**2 + (obs_path[:, 1] - cy)**2) - obs_path[:, 2]
                min_clear = float(np.min(dists_to_seg))
            else:
                min_clear = 9999.0
            
            near_clear_penalty = 1.0
            depth_pen = 1.0
        else:
            min_clear = 9999.0
            near_clear_penalty = 1.0
            depth_pen = 1.0
                
        min_clear = max(min_clear, 0)
        if min_clear < 15.0:
            continue

        # 갭 기둥(c1, c2) 자체 경로 간섭 검사:
        # 두 장애물이 배 진행방향과 평행(앞뒤)하게 서 있어서 앞 기둥이 배->중점(mid) 진입로를 가로막는 경우 즉시 배제
        col_pillar = False
        for pt in [c1, c2]:
            t_p = ((pt[0] - bx) * vx + (pt[1] - by) * vy) / seg2
            if 0.08 < t_p < 0.92:
                proj_x = bx + t_p * vx
                proj_y = by + t_p * vy
                d_p = math.hypot(pt[0] - proj_x, pt[1] - proj_y)
                if d_p < 30.0:
                    col_pillar = True
                    break
        if col_pillar:
            continue
            
        # 갭 선분(c1->c2)의 단위 벡터 및 게이트 법선 벡터
        v_gap = c2 - c1
        gap_w = math.hypot(v_gap[0], v_gap[1])
        u_gap = v_gap / (gap_w + 1e-6)
        n_gate = np.array([-u_gap[1], u_gap[0]])
        u_approach = rel / distm
        
        # 현재 배가 바라보는 헤딩(h_vec) 및 배에서 갭으로 들어가는 진입선(u_approach) 기준 상대너비
        # 배의 진행/진입 방향에 수직(직교)인 게이트일수록 상대너비 = 1.0 (100% 개방)
        # 배의 진행/진입 방향과 평행할수록 상대너비 = 0.0 (완전 닫힘)
        rel_width_h = abs(float(np.dot(n_gate, h_vec)))
        rel_width_app = abs(float(np.dot(n_gate, u_approach)))
        rel_width = min(rel_width_h, rel_width_app)
        
        # 체감 유효 통과 폭 (Effective Aperture Width)
        effective_width = gap_w * rel_width
        
        # 배의 반경이 25px(전폭 50px)이므로, 유효 통과폭이 42px 미만이거나
        # 선박 진행방향과 거의 평행(상대너비 0.22 미만, 약 13도 이내)한 통과 불가능 갭은 원천 배제
        if effective_width < 42.0 or rel_width < 0.22:
            continue
            
        clear_score = rel_width
        clear_factor = clear_score ** clear_exp
        
        # 갭 선분(c1->c2)과 현재 위치에서 목적지까지의 방향(gps_vec) 간의 수직도(Orthogonality) 계산
        perp_score = abs(gps_vec[0] * u_gap[1] - gps_vec[1] * u_gap[0])
        perp_factor = max(perp_score, 0.05) ** perp_exp
        
        # 선박과의 근접도 (Proximity to Boat): 배와 가까울수록 높은 점수 부여
        prox_score = min(1.0, 65.0 / max(distm, 35.0))
        prox_factor = max(prox_score, 0.05) ** prox_exp
        
        # 기준 위도(시작점-목적지 연결 직선: target_pos[1])에서 벗어난 정도에 따라 가운데로 복귀하려는 성질
        base_y = target_pos[1]
        boat_dev = abs(by - base_y)
        gap_dev = abs(my - base_y)
        center_closeness = math.exp(-((gap_dev / 160.0)**2))
        dev_ratio = min(boat_dev / 150.0, 2.0)
        eff_center_exp = center_exp * (0.4 + 0.8 * dev_ratio)
        center_factor = max(center_closeness, 0.05) ** eff_center_exp
        
        width_w = min(gap_w / 90, 1)
        
        sc = (heading_align**align_exp) * head_factor * (forward_proj**fwd_exp) * (lateral_full**0.5) * clear_factor * (width_w**width_exp) * depth_pen * near_clear_penalty * perp_factor * prox_factor * center_factor
        
        if sc > 0:
            term_align = float(heading_align**align_exp)
            term_head = float(head_factor)
            term_fwd = float(forward_proj**fwd_exp)
            term_clear = float(clear_factor)
            term_perp = float(perp_factor)
            term_prox = float(prox_factor)
            term_center = float(center_factor)
            
            valid_gaps.append({
                "pos": mid,
                "c1": c1.copy(),
                "c2": c2.copy(),
                "pair": (id1, id2),
                "score": sc,
                "factors": {
                    "Align": {"raw": float(heading_align), "w": float(align_exp)},
                    "Heading": {"raw": float(head_score), "w": float(heading_exp)},
                    "Forward": {"raw": float(forward_proj), "w": float(fwd_exp)},
                    "Clear": {"raw": float(clear_score), "w": float(clear_exp)},
                    "Perpend": {"raw": float(perp_score), "w": float(perp_exp)},
                    "Proxim": {"raw": float(prox_score), "w": float(prox_exp)},
                    "Center": {"raw": float(center_closeness), "w": float(eff_center_exp)}
                }
            })
            
    if not valid_gaps:
        return None
        
    valid_gaps.sort(key=lambda x: x["score"], reverse=True)
    best = valid_gaps[0]
    
    # 2번째, 3번째 웨이포인트 후보는 1순위(현재 웨이포인트) 및 앞선 후보와 장애물 쌍/위치가 중복되지 않도록 선별
    candidates = []
    selected_pairs = {tuple(sorted(best["pair"]))}
    selected_positions = [best["pos"]]
    
    for g in valid_gaps[1:]:
        pair_key = tuple(sorted(g["pair"]))
        # 1. 1순위 및 앞선 후보와 동일한 장애물 쌍 배제 (동일 갭 중복 배제)
        if pair_key in selected_pairs:
            continue
            
        # 2. 물리적 위치가 1순위 및 앞선 후보들과 너무 가까운 갭 배제 (최소 50px 이상 이격)
        too_close = False
        for spos in selected_positions:
            dp = g["pos"] - spos
            if math.hypot(dp[0], dp[1]) < 50.0:
                too_close = True
                break
        if too_close:
            continue
            
        candidates.append(g)
        selected_pairs.add(pair_key)
        selected_positions.append(g["pos"])
        if len(candidates) >= 2:
            break
            
    best["candidates"] = candidates
    
    # [순수 GUI 편의 기능 전용 데이터]
    # 주행 및 회피 알고리즘에는 일절 관여하지 않으며, 사용자가 화면에서 빨간 부표 사이의 
    # 모든 조합(2개->1갭, 3개->3갭 등 N C 2) 틈새 위치를 점으로 시각화하여 확인할 수 있도록 분리 전달
    gui_all_gaps = []
    for i in range(len(items)):
        c1 = items[i][2]
        for j in range(i + 1, len(items)):
            c2 = items[j][2]
            mid_pt = (c1 + c2) / 2.0
            gui_all_gaps.append({
                "pos": mid_pt.copy(),
                "c1": c1.copy(),
                "c2": c2.copy()
            })
            
    best["total_gaps_count"] = len(gui_all_gaps)
    best["all_gaps"] = gui_all_gaps
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