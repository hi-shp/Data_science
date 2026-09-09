import numpy as np
import math

def bezier_point(p0, p1, p2, t):
    return (1-t)*(1-t)*p0 + 2*(1-t)*t*p1 + t*t*p2

def build_bezier_path(p0, p1, p2, samples=40):
    ts = np.linspace(0, 1, samples)
    pts = [bezier_point(p0, p1, p2, t) for t in ts]
    return np.array(pts, dtype=np.float32)

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def cubic_bezier(p0, p1, p2, p3, n=90):
    t = np.linspace(0, 1, n)
    T = t[:, None]
    B = (1-T)**3 * p0 + 3*(1-T)**2*T*p1 + 3*(1-T)*T**2*p2 + T**3*p3
    return B

def make_bezier_path(boat_pos, boat_heading, goal, obstacles=None, boat_radius=25, min_clearance=35.0, boat_speed=0.0, start_tangent_fixed=False, next_goal=None):
    d = np.linalg.norm(goal - boat_pos)
    if d < 1:
        return None

    p0 = boat_pos.copy()
    p3 = goal.copy()

    v_to_goal = p3 - p0
    goal_angle = math.atan2(v_to_goal[1], v_to_goal[0])
    ang_diff = abs(wrap(goal_angle - boat_heading))
    u_goal = v_to_goal / d
    forward = np.array([math.cos(boat_heading), math.sin(boat_heading)])

    # 1. 장애물이 없는 목적지 직행 상황: 곡률을 대폭 줄여 목적지를 향해 직선에 가깝게 직행
    if obstacles is None or len(obstacles) == 0:
        if start_tangent_fixed:
            forward_dist = min(60.0, d * 0.40)
            p1 = boat_pos + forward * forward_dist
            v_goal = p3 - p1
            norm_v_goal = np.linalg.norm(v_goal)
            v_goal_n = np.zeros(2) if norm_v_goal < 1e-6 else v_goal / norm_v_goal
            p2 = p3 - v_goal_n * min(60.0, d * 0.38)
            return cubic_bezier(p0, p1, p2, p3, n=90)
        else:
            speed_ratio = np.clip(boat_speed / 80.0, 0.0, 1.0)
            forward_dist = min(35.0, d * 0.20) * max(0.15, math.cos(ang_diff * 0.5)) * (1.0 - 0.25 * speed_ratio)
            p1 = boat_pos + forward * forward_dist

            v_goal = p3 - p1
            norm_v_goal = np.linalg.norm(v_goal)
            v_goal_n = np.zeros(2) if norm_v_goal < 1e-6 else v_goal / norm_v_goal
            p2 = p3 - v_goal_n * min(35.0, d * 0.20)
            return cubic_bezier(p0, p1, p2, p3, n=90)

    # 2. 웨이포인트 추종 및 장애물 통과 구간: 선박 속도 및 각도 편차에 따른 선행 회전(Inward Lead) 및 관성 보정
    if start_tangent_fixed:
        # C1 연속성 유지를 위해 시작 접선 방향(forward)을 엄격히 고정
        forward_dist = min(45.0, d * 0.35)
        p1 = boat_pos + forward * forward_dist
    else:
        speed_ratio = np.clip(boat_speed / 80.0, 0.0, 1.0)
        lead_shrink = max(0.4, 1.0 - 0.50 * speed_ratio * math.sin(ang_diff * 0.5))
        forward_dist = min(45.0, d * 0.35) * lead_shrink

        # P1 방향을 목표 방향 안쪽으로 미리 편향하여 조기 선회 유도 (Inward Lead Vector)
        blend = min(0.35, 0.45 * speed_ratio * math.sin(ang_diff * 0.5))
        lead_dir = (1.0 - blend) * forward + blend * u_goal
        norm_lead = np.linalg.norm(lead_dir)
        lead_dir = forward if norm_lead < 1e-6 else lead_dir / norm_lead

        p1 = boat_pos + lead_dir * forward_dist

    # P2 제어점: 다음 웨이포인트(next_goal) 방향으로 접선 방향을 미리 비스듬히 유도하여 C1 연속 S자 곡선 생성
    v_goal = p3 - p1
    norm_v_goal = np.linalg.norm(v_goal)
    v_goal_n = np.zeros(2) if norm_v_goal < 1e-6 else v_goal / norm_v_goal

    if next_goal is not None:
        v_next = next_goal - p3
        norm_v_next = np.linalg.norm(v_next)
        if norm_v_next > 1e-3:
            u_next = v_next / norm_v_next
            # 진입 방향(v_goal_n)과 다음 목표 방향(u_next)을 완만하게 블렌딩 (가중치 35%)
            t_blend = 0.35
            t_mid = (1.0 - t_blend) * v_goal_n + t_blend * u_next
            norm_t = np.linalg.norm(t_mid)
            if norm_t > 1e-6:
                v_goal_n = t_mid / norm_t

    p2 = p3 - v_goal_n * min(60.0, d * 0.35)

    nominal_pts = cubic_bezier(p0, p1, p2, p3, n=30)
    u_dir = (p3 - p0) / d
    n_dir = np.array([-u_dir[1], u_dir[0]], dtype=np.float32)
    
    shift_p1 = np.zeros(2, dtype=np.float32)
    shift_p2 = np.zeros(2, dtype=np.float32)
    
    if len(obstacles) > 0:
        obs_pos_all = obstacles[:, :2]  # (M, 2)
        obs_rad_all = obstacles[:, 2]   # (M,)
        
        # (30, M) 거리 행렬을 한 번에 계산
        diff_x = nominal_pts[:, 0:1] - obs_pos_all[:, 0]  # (30, M)
        diff_y = nominal_pts[:, 1:2] - obs_pos_all[:, 1]  # (30, M)
        dists_all = np.sqrt(diff_x**2 + diff_y**2)         # (30, M)
        min_idx_all = np.argmin(dists_all, axis=0)          # (M,)
        min_dist_all = dists_all[min_idx_all, np.arange(len(obs_rad_all))]  # (M,)
        
        # 먼 거리(약 125px)부터 서서히 감지하여 특정 지점에서 휙 꺾이지 않고 멀리서부터 부드러운 호를 그리며 조기 우회
        far_margins = obs_rad_all + boat_radius + 80.0   # 약 122px (원거리 감지 범위)
        safe_margins = obs_rad_all + boat_radius + 35.0  # 약 77px (안전 마진)
        
        active = min_dist_all < far_margins
        if np.any(active):
            active_idx = np.where(active)[0]
            for k in active_idx:
                min_idx = min_idx_all[k]
                t_idx = min_idx / 29.0
                if start_tangent_fixed and t_idx < 0.2:
                    continue
                
                d_cur = min_dist_all[k]
                # 122px부터 0으로 시작하여 거리가 좁혀질수록 2차 곡선으로 점진적이고 매끄럽게 반발력 증가
                ratio = float(np.clip((far_margins[k] - d_cur) / (far_margins[k] - safe_margins[k] + 12.0), 0.0, 1.0))
                push_mag = 52.0 * (ratio ** 1.6)
                if d_cur < safe_margins[k]:
                    push_mag += (safe_margins[k] - d_cur) * 1.1
                push_mag = min(130.0, push_mag)
                
                pt = nominal_pts[min_idx]
                obs_pos = obs_pos_all[k]
                vec_from_obs = pt - obs_pos
                norm_vec = math.hypot(vec_from_obs[0], vec_from_obs[1])
                if norm_vec > 1e-4:
                    push_dir = vec_from_obs / norm_vec
                else:
                    side = (obs_pos[0] - p0[0]) * n_dir[0] + (obs_pos[1] - p0[1]) * n_dir[1]
                    push_dir = -n_dir if side >= 0 else n_dir
                    
                w1 = max(0.2, 1.0 - t_idx * 0.7)
                w2 = max(0.2, t_idx * 0.7 + 0.3)
                
                if not start_tangent_fixed:
                    shift_p1 += push_dir * (push_mag * w1)
                shift_p2 += push_dir * (push_mag * w2)
            
    p1 = p1 + shift_p1
    p2 = p2 + shift_p2

    return cubic_bezier(p0, p1, p2, p3, n=90)

def pure_pursuit(path, boat_pos, lookahead=70):
    if path is None or len(path) == 0:
        return None
    dists = np.sqrt(np.sum((path - boat_pos)**2, axis=1))
    far = np.where(dists > lookahead)[0]
    if len(far) > 0:
        return path[far[0]]
    return path[-1]

def find_pp_target(path, pos, L=80):
    for i in range(len(path)-1):
        if np.linalg.norm(path[i]-pos) >= L:
            return path[i]
    return path[-1]