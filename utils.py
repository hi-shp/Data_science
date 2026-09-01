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

def make_bezier_path(boat_pos, boat_heading, goal, obstacles=None, boat_radius=25, min_clearance=35.0):
    d = np.linalg.norm(goal - boat_pos)
    if d < 1:
        return None

    p0 = boat_pos.copy()
    p3 = goal.copy()

    # 각도 편차 계산
    v_to_goal = p3 - p0
    goal_angle = math.atan2(v_to_goal[1], v_to_goal[0])
    ang_diff = abs(wrap(goal_angle - boat_heading))
    
    # 전방 제어점 거리를 기존(65)보다 넉넉하게 확장(최대 110px)하여 부드럽고 풍성한 선회 곡률 형성
    forward_dist = min(110.0, d * 0.45)
    
    forward = np.array([math.cos(boat_heading), math.sin(boat_heading)])
    p1 = boat_pos + forward * forward_dist

    v_goal = p3 - p1
    norm_v_goal = np.linalg.norm(v_goal)
    v_goal_n = np.zeros(2) if norm_v_goal < 1e-6 else v_goal / norm_v_goal
        
    p2 = p3 - v_goal_n * min(100.0, d * 0.40)

    # 장애물 회피를 위한 베지어 제어점(p1, p2) 외측 굴곡/우회 처리 (Obstacle-avoiding Bezier curve deflection)
    if obstacles is not None and len(obstacles) > 0:
        nominal_pts = cubic_bezier(p0, p1, p2, p3, n=30)
        
        u_dir = (p3 - p0) / d
        n_dir = np.array([-u_dir[1], u_dir[0]], dtype=np.float32)
        
        shift_p1 = np.zeros(2, dtype=np.float32)
        shift_p2 = np.zeros(2, dtype=np.float32)
        
        for (ox, oy, orad) in obstacles:
            obs_pos = np.array([ox, oy], dtype=np.float32)
            vec_obs = obs_pos - p0
            proj = np.dot(vec_obs, u_dir)
            
            # 장애물이 출발점과 도착점 전방 구간에 위치하는 경우
            if 0 < proj < d:
                dists = np.linalg.norm(nominal_pts - obs_pos, axis=1)
                min_idx = np.argmin(dists)
                min_dist = dists[min_idx]
                t_idx = min_idx / 29.0
                
                safe_margin = orad + boat_radius + min_clearance
                encroach = safe_margin - min_dist
                
                if encroach > 0:
                    side = np.dot(obs_pos - p0, n_dir)
                    push_dir = -n_dir if side >= 0 else n_dir
                    
                    push_mag = min(130.0, encroach * 1.6)
                    
                    w1 = max(0.2, 1.0 - t_idx * 0.7)
                    w2 = max(0.2, t_idx * 0.7 + 0.3)
                    
                    shift_p1 += push_dir * (push_mag * w1)
                    shift_p2 += push_dir * (push_mag * w2)
                    
        p1 = p1 + shift_p1
        p2 = p2 + shift_p2

    return cubic_bezier(p0, p1, p2, p3, n=90)

def pure_pursuit(path, boat_pos, lookahead=70):
    if path is None:
        return None

    for i in range(len(path)-1):
        p = path[i]
        if np.linalg.norm(p - boat_pos) > lookahead:
            return p
    
    return path[-1]

def find_pp_target(path, pos, L=80):
    for i in range(len(path)-1):
        if np.linalg.norm(path[i]-pos) >= L:
            return path[i]
    return path[-1]