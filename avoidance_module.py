import numpy as np
import math


def repulsive_force(lidar, rel_cos, rel_sin, boat_heading, lidar_range, k_rep, d0, v_factor):
    cosH = math.cos(boat_heading)
    sinH = math.sin(boat_heading)

    # 회전 변환
    rx = cosH * rel_cos - sinH * rel_sin
    ry = sinH * rel_cos + cosH * rel_sin

    # 0 방지
    d = np.clip(lidar, 1e-3, lidar_range)

    # inverse-square hybrid repulsion
    # 멀리선 완만, 가까울수록 급격
    w = np.clip((d0 - d) / d0, 0, 1)
    rep = k_rep * w * (1.0 / (d*d))

    # 속도 기반 강화
    rep *= (1.0 + v_factor)

    R = np.array([
        -np.sum(rep * rx),
        -np.sum(rep * ry)
    ], dtype=np.float32)

    return R


def detect_gap(lidar, rel_angles):
    n = len(lidar)
    far = lidar.max()

    # threshold: 전체 거리 중 상위 30% 이상만 gap 후보
    threshold = far * 0.7
    mask = lidar >= threshold

    gap_indices = np.where(mask)[0]
    if len(gap_indices) == 0:
        return None

    clusters = []
    start = gap_indices[0]
    prev = start

    for idx in gap_indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            clusters.append((start, prev))
            start = idx
            prev = idx
    clusters.append((start, prev))

    best = None
    best_len = -1
    for a, b in clusters:
        L = b - a + 1
        if L > best_len:
            best_len = L
            best = (a, b)

    a, b = best
    c = (a + b) // 2
    return rel_angles[c]


def gap_vector(boat_heading, gap_angle, w_gap):
    if gap_angle is None:
        return np.zeros(2, dtype=np.float32)

    ang = boat_heading + gap_angle
    return np.array([
        math.cos(ang),
        math.sin(ang)
    ]) * w_gap


def goal_vector(gps_heading, w_goal):
    return np.array([
        math.cos(gps_heading),
        math.sin(gps_heading)
    ]) * w_goal


def combine_heading(A, R, G):
    V = A + R + G
    return math.atan2(V[1], V[0])


def compute_safe_heading(boat_heading,
                         gps_heading,
                         lidar,
                         rel_angles,
                         rel_cos,
                         rel_sin,
                         lidar_range,
                         v_boat,
                         k_goal=15.0,
                         k_rep=1.5,
                         k_gap=6.0,
                         d0=250.0):

    A = goal_vector(gps_heading, k_goal)

    R = repulsive_force(
        lidar=lidar,
        rel_cos=rel_cos,
        rel_sin=rel_sin,
        boat_heading=boat_heading,
        lidar_range=lidar_range,
        k_rep=k_rep,
        d0=d0,
        v_factor=v_boat
    )

    gap_angle = detect_gap(lidar, rel_angles)
    G = gap_vector(boat_heading, gap_angle, k_gap)

    heading = combine_heading(A, R, G)
    return heading
