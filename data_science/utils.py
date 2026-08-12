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
    forward = np.array([math.cos(boat_heading), math.sin(boat_heading)])
    p1 = boat_pos + forward * min(120, d * 0.4)

    p3 = goal.copy()
    v_goal = p3 - p1
    norm_v_goal = np.linalg.norm(v_goal)
    if norm_v_goal < 1e-6:
        v_goal_n = np.zeros(2)
    else:
        v_goal_n = v_goal / norm_v_goal
        
    p2 = p3 - v_goal_n * min(120, d * 0.4)

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