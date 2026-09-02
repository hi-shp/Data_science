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

def make_bezier_path(boat_pos, boat_heading, goal):
    d = np.linalg.norm(goal - boat_pos)
    if d < 1:
        return None

    p0 = boat_pos.copy()
    p3 = goal.copy()

    # 각도 편차에 따른 전방 제어점 거리 축소 (관성으로 인한 외측 쏠림 방지)
    v_to_goal = p3 - p0
    goal_angle = math.atan2(v_to_goal[1], v_to_goal[0])
    ang_diff = abs(wrap(goal_angle - boat_heading))
    
    # 꺾이는 각도가 클수록 전방 돌출을 줄이고 곡선 시작점을 당겨 즉각적인 회전 유도
    forward_dist = min(65, d * 0.35) * max(0.25, math.cos(ang_diff * 0.5))
    
    forward = np.array([math.cos(boat_heading), math.sin(boat_heading)])
    p1 = boat_pos + forward * forward_dist

    v_goal = p3 - p1
    norm_v_goal = np.linalg.norm(v_goal)
    v_goal_n = np.zeros(2) if norm_v_goal < 1e-6 else v_goal / norm_v_goal
        
    p2 = p3 - v_goal_n * min(65, d * 0.35)

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