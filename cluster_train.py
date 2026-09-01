"""
cluster_train.py - 이멀전시 히스테리시스 & 갭 우선순위 가중치 포함 24코어 병렬 강화학습
"""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import multiprocessing as mp
import numpy as np
import math
import copy
import time
import json
import random
from perception import lidar_hits_np, update_grid, extract_clusters_from_grid, match_clusters
from navigation import target_is_clear, reactive_avoidance
from utils import wrap, make_bezier_path, pure_pursuit
from config import WIDTH, HEIGHT, GRID, GRID_W, GRID_H

def parameterized_find_gap(clusters, ids, boat_pos, boat_heading, target_pos, visited, grid, obstacles,
                           align_exp=5.0, fwd_exp=3.0, clear_exp=1.5, width_exp=0.2, cluster_pen_w=0.5):
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
        cluster_pen = math.exp(-cluster_pen_w * cnt)
        
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
        
        sc = (heading_align**align_exp) * (forward_proj**fwd_exp) * (lateral_full**0.5) * (path_clear**clear_exp) * (width_w**width_exp) * cluster_pen * depth_pen
        
        if sc > best_sc:
            best_sc = sc
            best = {"pos": mid, "c1": c1.copy(), "c2": c2.copy(), "pair": (id1, id2), "score": sc}
            
    return best

class FastBoatSim:
    def __init__(self):
        self.w = WIDTH
        self.h = HEIGHT
        self.sim_h = 630
        self.dt = 0.04
        self.lidar_beams = 180
        self.lidar_range = 350
        self.rel_angles = np.linspace(-np.pi, np.pi, self.lidar_beams, endpoint=False)
        self.mass = 20
        self.inertia = 6
        self.drag = 0.38
        self.rot_drag = 0.55
        self.boat_radius = 25
        self.obs_n = 80
        self.obs_r = 17
        self.min_obs = 120
        self.grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.boat_pos = np.array([65, self.sim_h/2], dtype=np.float32)
        self.boat_vel = np.zeros(2)
        self.boat_ang_vel = 0.0
        self.target = np.array([self.w - 100, self.sim_h/2], dtype=np.float32)
        
        obs = []
        t = 0
        while len(obs) < self.obs_n and t < 5000:
            t += 1
            x = random.randint(300, self.w - 300)
            y = random.randint(30, self.sim_h - 30)
            p = np.array([x, y])
            if np.linalg.norm(p - self.target) < 180: continue
            if np.linalg.norm(p - self.boat_pos) < 180: continue
            ok = True
            for (ox, oy, r) in obs:
                if np.linalg.norm(p - np.array([ox, oy])) < self.min_obs:
                    ok = False
                    break
            if ok:
                obs.append((x, y, self.obs_r))
                
        self.obstacles = np.array(obs, dtype=np.float32)
        self.dynamic_obstacles = self.obstacles.copy()
        dx = self.target[0] - self.boat_pos[0]
        dy = self.target[1] - self.boat_pos[1]
        self.boat_heading = math.atan2(dy, dx)
        
        self.grid[:] = 0
        self.clusters = []
        self.cluster_ids = []
        self.current_wp = None
        self.next_wp = None
        self.visited = set()
        self.wp_check_timer = 0
        self.steer_timer = 0
        self.path_timer = 0
        self.bezier_path = None
        self.next_bezier_path = None
        self.pursuit_target = None
        self.next_pursuit_target = None
        self.prev_steer = 0.0
        self.current_fwd = 0.0
        self.frame = 0
        self.emergency_mode = False
        self.emergency_cooldown = 0

    def update_dynamic_obstacles(self):
        self.dynamic_obstacles = self.obstacles.copy()
        for i in range(len(self.obstacles)):
            ox, oy, r = self.obstacles[i]
            sway_x = math.sin(self.frame * 0.03 + oy * 0.1) * (r * 0.2)
            sway_y = math.cos(self.frame * 0.04 + ox * 0.1) * (r * 0.2)
            self.dynamic_obstacles[i, 0] = ox + sway_x
            self.dynamic_obstacles[i, 1] = oy + sway_y

    def collide(self):
        ox = self.dynamic_obstacles[:, 0]
        oy = self.dynamic_obstacles[:, 1]
        rr = self.dynamic_obstacles[:, 2] + self.boat_radius
        dx = ox - self.boat_pos[0]
        dy = oy - self.boat_pos[1]
        hit = np.any(dx*dx + dy*dy <= rr*rr)
        wall = (self.boat_pos[0] <= 0 or self.boat_pos[0] >= self.w or
                self.boat_pos[1] <= 0 or self.boat_pos[1] >= self.sim_h)
        return hit or wall

    def validate_wp_grid(self):
        if self.current_wp is None: return
        self.wp_check_timer += self.dt
        if self.wp_check_timer < 0.05: return
        self.wp_check_timer = 0
        wp = self.current_wp["pos"]; pair = self.current_wp["pair"]
        gx = int(wp[0] // GRID); gy = int(wp[1] // GRID); rad = int(35 // GRID)
        for yy in range(max(0, gy - rad), min(GRID_H, gy + rad + 1)):
            for xx in range(max(0, gx - rad), min(GRID_W, gx + rad + 1)):
                if self.grid[yy, xx] >= 3:
                    self.visited.add(pair); self.visited.add((pair[1], pair[0]))
                    self.current_wp = None; return

    def validate_wp_obstacle_5x5(self):
        if self.current_wp is None: return
        wp = self.current_wp["pos"]
        gx = int(wp[0] // GRID); gy = int(wp[1] // GRID)
        xs = range(gx - 2, gx + 3); ys = range(gy - 2, gy + 3)
        ox = self.dynamic_obstacles[:, 0]; oy = self.dynamic_obstacles[:, 1]; rr = self.dynamic_obstacles[:, 2]
        for yy in ys:
            for xx in xs:
                if 0 <= xx < GRID_W and 0 <= yy < GRID_H:
                    cx = xx * GRID + GRID * 0.5; cy = yy * GRID + GRID * 0.5
                    dx = ox - cx; dy = oy - cy
                    hit = np.any(dx*dx + dy*dy <= rr*rr)
                    if hit:
                        p = self.current_wp["pair"]
                        self.visited.add(p); self.visited.add((p[1], p[0]))
                        self.current_wp = None; return

_worker_sim = None

def worker_init():
    global _worker_sim
    _worker_sim = FastBoatSim()

def run_sim_task(args):
    global _worker_sim
    seed, params = args
    sim = _worker_sim
    sim.reset(seed)
    
    steer_gain = params['steer_gain']
    steer_alpha = params['steer_alpha']
    mom_coeff = params['mom_coeff']
    pwm_rng = params['pwm_rng']
    avoid_normal = params['avoid_normal']
    avoid_em = params['avoid_em']
    clear_margin = params['clear_margin']
    
    # 이멀전시 히스테리시스 파라미터
    em_enter = params.get('em_enter', 75.0)
    em_exit = params.get('em_exit', 115.0)
    em_hold_frames = int(params.get('em_hold_frames', 12))
    
    # 갭 우선순위 가중치
    align_exp = params.get('align_exp', 5.0)
    fwd_exp = params.get('fwd_exp', 3.0)
    clear_exp = params.get('clear_exp', 1.5)
    width_exp = params.get('width_exp', 0.2)
    cluster_pen_w = params.get('cluster_pen_w', 0.5)
    wp_switch_thresh = params.get('wp_switch_thresh', 1.15)
    
    wp_arrive = 25.0
    max_frames = 2600

    for frame in range(max_frames):
        sim.frame += 1
        sim.update_dynamic_obstacles()

        dists, hits = lidar_hits_np(
            sim.boat_pos, sim.boat_heading,
            sim.rel_angles, sim.dynamic_obstacles,
            sim.lidar_range
        )

        update_grid(sim.grid, hits)
        sim.grid *= 0.945

        if frame % 2 == 0:
            new_c = extract_clusters_from_grid(sim.grid)
            sim.clusters, sim.cluster_ids = match_clusters(
                sim.clusters, sim.cluster_ids, new_c
            )
            sim.validate_wp_grid()
            sim.validate_wp_obstacle_5x5()

        if target_is_clear(sim.boat_pos, sim.target, sim.dynamic_obstacles, boat_radius=25 + clear_margin):
            sim.current_wp = None
            sim.next_wp = None
            new_wp = None
        else:
            new_wp = parameterized_find_gap(
                sim.clusters, sim.cluster_ids,
                sim.boat_pos, sim.boat_heading,
                sim.target, sim.visited,
                sim.grid, sim.dynamic_obstacles,
                align_exp, fwd_exp, clear_exp, width_exp, cluster_pen_w
            )

        if sim.current_wp is not None:
            should_clear = False
            vec_to_wp = sim.current_wp["pos"] - sim.boat_pos
            dnow = np.linalg.norm(vec_to_wp)
            if dnow < wp_arrive:
                should_clear = True
            wp_angle = math.atan2(vec_to_wp[1], vec_to_wp[0])
            angle_diff = abs(wrap(wp_angle - sim.boat_heading))
            if angle_diff > np.pi / 2 and dnow < 60:
                should_clear = True
            if should_clear:
                p = sim.current_wp["pair"]
                sim.visited.add(p)
                sim.visited.add((p[1], p[0]))
                sim.current_wp = None

        if new_wp is not None:
            if sim.current_wp is None:
                sim.current_wp = new_wp
            else:
                dist_to_curr = np.linalg.norm(sim.current_wp["pos"] - sim.boat_pos)
                if dist_to_curr > 80:
                    if new_wp["score"] > sim.current_wp["score"] * wp_switch_thresh:
                        sim.current_wp = new_wp

        if sim.current_wp is not None:
            temp_visited = sim.visited.copy()
            temp_visited.add(sim.current_wp["pair"])
            temp_visited.add((sim.current_wp["pair"][1], sim.current_wp["pair"][0]))
            vec = sim.current_wp["pos"] - sim.boat_pos
            next_head = math.atan2(vec[1], vec[0])
            sim.next_wp = parameterized_find_gap(
                sim.clusters, sim.cluster_ids,
                sim.current_wp["pos"], next_head,
                sim.target, temp_visited,
                sim.grid, sim.dynamic_obstacles,
                align_exp, fwd_exp, clear_exp, width_exp, cluster_pen_w
            )
        else:
            sim.next_wp = None

        sim.path_timer += sim.dt
        if sim.path_timer >= 0.01:
            sim.path_timer = 0
            if sim.current_wp is None:
                goal = sim.target
            else:
                goal = sim.current_wp["pos"]
            sim.bezier_path = make_bezier_path(sim.boat_pos, sim.boat_heading, goal)
            if sim.bezier_path is not None:
                sim.pursuit_target = pure_pursuit(sim.bezier_path, sim.boat_pos, lookahead=52)
            if sim.current_wp is not None and sim.next_wp is not None:
                vec = sim.current_wp["pos"] - sim.boat_pos
                next_start_head = math.atan2(vec[1], vec[0])
                sim.next_bezier_path = make_bezier_path(sim.current_wp["pos"], next_start_head, sim.next_wp["pos"])
                if sim.next_bezier_path is not None:
                    sim.next_pursuit_target = pure_pursuit(sim.next_bezier_path, sim.current_wp["pos"], lookahead=52)
            else:
                sim.next_bezier_path = None
                sim.next_pursuit_target = None

        visual_target = sim.pursuit_target
        if sim.current_wp is not None and sim.next_pursuit_target is not None and sim.pursuit_target is not None:
            dist_to_wp = np.linalg.norm(sim.current_wp["pos"] - sim.boat_pos)
            if dist_to_wp < 75:
                sim.pursuit_target = sim.next_pursuit_target

        # 이멀전시 히스테리시스 판정
        center_idx = sim.lidar_beams // 2
        span = sim.lidar_beams // 12
        front_dists = dists[center_idx - span : center_idx + span]
        min_front_dist = np.min(front_dists)
        
        if min_front_dist < em_enter:
            sim.emergency_mode = True
            sim.emergency_cooldown = em_hold_frames
        elif sim.emergency_mode:
            sim.emergency_cooldown -= 1
            if min_front_dist > em_exit and sim.emergency_cooldown <= 0:
                sim.emergency_mode = False
        
        is_emergency = sim.emergency_mode
        
        if sim.pursuit_target is not None:
            px, py = sim.pursuit_target
            heading_target = math.atan2(py - sim.boat_pos[1], px - sim.boat_pos[0])
            heading_error = wrap(heading_target - sim.boat_heading)
            steer_raw = heading_error * steer_gain
            steer_f = steer_alpha * steer_raw + (1 - steer_alpha) * sim.prev_steer
            sim.prev_steer = steer_f
            
            avoid = reactive_avoidance(dists, sim.rel_angles)
            avoid_multiplier = avoid_em if is_emergency else avoid_normal
            steer = float(np.clip(steer_f + avoid_multiplier * avoid, -1, 1))
        else:
            steer = 0

        dead = 0.02
        st = steer if abs(steer) >= dead else 0
        mid = 1500
        m = np.log1p(3 * abs(st)) / np.log(4)
        d = m * pwm_rng
        if st >= 0: L = mid - d; R = mid + d
        else: L = mid + d; R = mid - d
        L = int(np.clip(L, 1100, 1900))
        R = int(np.clip(R, 1100, 1900))

        tL = L * 10
        tR = R * 10
        target_fwd = (tL + tR) / 9.0
        if is_emergency:
            target_fwd = (tL + tR) / 22.0  # 완전 감속 대신 선회 추진력을 살려 완벽 회피
        if not hasattr(sim, 'current_fwd'):
            sim.current_fwd = 0.0
        sim.current_fwd = sim.current_fwd * 0.95 + target_fwd * 0.05
        
        mom = (tR - tL) * mom_coeff
        hv = np.array([math.cos(sim.boat_heading), math.sin(sim.boat_heading)])
        acc = sim.current_fwd / sim.mass
        vel_norm = np.linalg.norm(sim.boat_vel)
        drag = -sim.drag * vel_norm * sim.boat_vel if vel_norm > 0 else np.zeros(2)
        
        sim.boat_vel += (acc * hv + drag) * sim.dt
        sim.boat_pos += sim.boat_vel * sim.dt
        
        ang_acc = (mom - sim.rot_drag * sim.boat_ang_vel) / sim.inertia
        sim.boat_ang_vel += ang_acc * sim.dt
        sim.boat_ang_vel *= 0.84
        sim.boat_heading += sim.boat_ang_vel * sim.dt

        dist_to_target = np.linalg.norm(sim.target - sim.boat_pos)
        if sim.collide():
            return 'collide', frame
        if dist_to_target < 70:
            return 'goal', frame

    return 'timeout', max_frames

def mutate_params(base, scale=0.05):
    p = copy.deepcopy(base)
    # 조타 및 물리
    p['steer_gain'] = float(np.clip(p['steer_gain'] + np.random.normal(0, 0.02 * scale * 10), 0.70, 0.92))
    p['steer_alpha'] = float(np.clip(p['steer_alpha'] + np.random.normal(0, 0.015 * scale * 10), 0.30, 0.45))
    p['mom_coeff'] = float(np.clip(p['mom_coeff'] + np.random.normal(0, 0.0002 * scale * 10), 0.0060, 0.0078))
    p['pwm_rng'] = float(np.clip(p['pwm_rng'] + np.random.normal(0, 5 * scale * 10), 250, 300))
    p['avoid_normal'] = float(np.clip(p['avoid_normal'] + np.random.normal(0, 0.0015 * scale * 10), 0.016, 0.030))
    p['avoid_em'] = float(np.clip(p['avoid_em'] + np.random.normal(0, 0.008 * scale * 10), 0.08, 0.16))
    p['clear_margin'] = float(np.clip(p['clear_margin'] + np.random.normal(0, 0.4 * scale * 10), 1.5, 4.0))
    
    # 이멀전시 히스테리시스
    p['em_enter'] = float(np.clip(p['em_enter'] + np.random.normal(0, 2.0 * scale * 10), 65.0, 90.0))
    p['em_exit'] = float(np.clip(p['em_exit'] + np.random.normal(0, 3.0 * scale * 10), 100.0, 140.0))
    p['em_hold_frames'] = int(np.clip(p['em_hold_frames'] + int(np.random.normal(0, 2 * scale * 10)), 8, 25))
    
    # 갭 가중치
    p['align_exp'] = float(np.clip(p['align_exp'] + np.random.normal(0, 0.3 * scale * 10), 3.0, 7.0))
    p['fwd_exp'] = float(np.clip(p['fwd_exp'] + np.random.normal(0, 0.2 * scale * 10), 2.0, 4.5))
    p['clear_exp'] = float(np.clip(p['clear_exp'] + np.random.normal(0, 0.15 * scale * 10), 1.0, 2.5))
    p['width_exp'] = float(np.clip(p['width_exp'] + np.random.normal(0, 0.05 * scale * 10), 0.1, 0.4))
    p['wp_switch_thresh'] = float(np.clip(p['wp_switch_thresh'] + np.random.normal(0, 0.03 * scale * 10), 1.08, 1.35))
    return p

def main():
    n_workers = min(12, os.cpu_count() or 4)
    print("==================================================================", flush=True)
    print(f"  24코어 이멀전시 히스테리시스 & 갭 가중치 강화학습 (Workers: {n_workers:2d})  ", flush=True)
    print("==================================================================", flush=True)
    print("  • 이멀전시 진입/해제 히스테리시스 및 최소 쿨다운(Hold) 도입", flush=True)
    print("  • 갭(Waypoint) 선정 우선순위 가중치 5종 통합 최적화", flush=True)
    print("──────────────────────────────────────────────────────────────────", flush=True)

    pool = mp.Pool(processes=n_workers, initializer=worker_init)

    best_params = {
        'steer_gain': 0.7752,
        'steer_alpha': 0.3515,
        'mom_coeff': 0.00665,
        'pwm_rng': 270.36,
        'avoid_normal': 0.019,
        'avoid_em': 0.11,
        'clear_margin': 2.15,
        'em_enter': 78.0,
        'em_exit': 118.0,
        'em_hold_frames': 14,
        'align_exp': 5.0,
        'fwd_exp': 3.0,
        'clear_exp': 1.5,
        'width_exp': 0.2,
        'cluster_pen_w': 0.5,
        'wp_switch_thresh': 1.15
    }

    best_rate = 86.0
    generation = 0

    try:
        while True:
            generation += 1
            seed_offset = generation * 1000 + 100000
            
            if generation == 1:
                candidate = copy.deepcopy(best_params)
            else:
                candidate = mutate_params(best_params, scale=0.04 + 0.03 * (generation % 3 == 0))

            t0 = time.time()
            from tqdm import tqdm
            tasks_stage1 = [(seed_offset + i, candidate) for i in range(40)]
            
            goals1 = 0
            collisions1 = 0
            
            # 1단계 TQDM 바
            pbar1 = tqdm(pool.imap_unordered(run_sim_task, tasks_stage1, chunksize=2), total=40, desc=f"Gen {generation:03d} Stage 1", leave=False)
            for r, f in pbar1:
                if r == 'goal': goals1 += 1
                elif r == 'collide': collisions1 += 1
                pbar1.set_postfix({'goals': goals1, 'col': collisions1})
                
            t_elapsed1 = time.time() - t0
            rate1 = goals1 / 40.0 * 100.0

            is_elite = (rate1 >= 90.0)
            tag = "ELITE 후보!" if is_elite else ""
            print(f"Gen {generation:03d} | [1단계 40회] 도달: {goals1:2d}/40 ({rate1:.1f}%) | 충돌: {collisions1:2d} | {t_elapsed1:.1f}초 {tag}", flush=True)

            if is_elite or generation == 1:
                t0_2 = time.time()
                from tqdm import tqdm
                tasks_stage2 = [(seed_offset + 500 + i, candidate) for i in range(100)]
                
                goals2 = 0
                collisions2 = 0
                timeouts2 = 0
                
                # 2단계 TQDM 바
                pbar2 = tqdm(pool.imap_unordered(run_sim_task, tasks_stage2, chunksize=4), total=100, desc="Stage 2", leave=False)
                for r, f in pbar2:
                    if r == 'goal': goals2 += 1
                    elif r == 'collide': collisions2 += 1
                    elif r == 'timeout': timeouts2 += 1
                    pbar2.set_postfix({'goals': goals2, 'col': collisions2})
                    
                t_elapsed2 = time.time() - t0_2
                rate2 = goals2 / 100.0 * 100.0

                print(f"   ↳ [2단계 100회 정밀] 도달: {goals2:3d}/100 ({rate2:.1f}%) | 충돌: {collisions2:2d} | 타임아웃: {timeouts2:2d} | {t_elapsed2:.1f}초", flush=True)

                if rate2 >= best_rate:
                    best_rate = rate2
                    best_params = copy.deepcopy(candidate)
                    print(f"   ★ [최고 성능 갱신!] 도달률: {best_rate:.1f}% | params: {best_params}", flush=True)

                if goals2 == 100 or rate2 >= 95.0:
                    print("\n" + "=" * 66, flush=True)
                    print(f"  축하합니다! 100회 무작위 맵 {rate2:.1f}% 초고도 무충돌 완주 달성! ", flush=True)
                    print("=" * 66, flush=True)
                    print(f"  최종 최적 파라미터 세트:", flush=True)
                    for k, v in best_params.items():
                        print(f"    • {k:18s}: {v}")
                    print("──────────────────────────────────────────────────────────────────", flush=True)
                    
                    with open("best_learned_params.json", "w") as f:
                        json.dump(best_params, f, indent=2)
                    print("[완료] best_learned_params.json 파일 저장 완료.", flush=True)
                    break

    finally:
        pool.close()
        pool.join()

if __name__ == "__main__":
    main()
