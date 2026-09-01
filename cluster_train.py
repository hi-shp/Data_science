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

    def step_sim(self, params):
        self.frame += 1
        self.update_dynamic_obstacles()

        steer_gain = params['steer_gain']
        steer_alpha = params['steer_alpha']
        mom_coeff = params['mom_coeff']
        pwm_rng = params['pwm_rng']
        avoid_normal = params['avoid_normal']
        avoid_em = params['avoid_em']
        clear_margin = params['clear_margin']
        em_enter = params.get('em_enter', 115.0)
        em_exit = params.get('em_exit', 160.0)
        em_hold_frames = int(params.get('em_hold_frames', 18))
        
        align_exp = params.get('align_exp', 5.0)
        fwd_exp = params.get('fwd_exp', 3.0)
        clear_exp = params.get('clear_exp', 1.5)
        width_exp = params.get('width_exp', 0.2)
        cluster_pen_w = params.get('cluster_pen_w', 0.5)
        wp_switch_thresh = params.get('wp_switch_thresh', 1.15)

        dists, hits = lidar_hits_np(
            self.boat_pos, self.boat_heading,
            self.rel_angles, self.dynamic_obstacles,
            self.lidar_range
        )

        update_grid(self.grid, hits)
        self.grid *= 0.945

        if self.frame % 2 == 0:
            new_c = extract_clusters_from_grid(self.grid)
            self.clusters, self.cluster_ids = match_clusters(
                self.clusters, self.cluster_ids, new_c
            )

        if target_is_clear(self.boat_pos, self.target, self.dynamic_obstacles, boat_radius=25 + clear_margin):
            self.current_wp = None
            self.next_wp = None
            new_wp = None
        else:
            new_wp = parameterized_find_gap(
                self.clusters, self.cluster_ids,
                self.boat_pos, self.boat_heading,
                self.target, self.visited,
                self.grid, self.dynamic_obstacles,
                align_exp, fwd_exp, clear_exp, width_exp, cluster_pen_w
            )

        if self.current_wp is not None:
            vec_to_wp = self.current_wp["pos"] - self.boat_pos
            dnow = np.linalg.norm(vec_to_wp)
            should_clear = False
            if dnow < 25.0: should_clear = True
            wp_angle = math.atan2(vec_to_wp[1], vec_to_wp[0])
            if abs(wrap(wp_angle - self.boat_heading)) > np.pi / 2 and dnow < 60:
                should_clear = True
            if should_clear:
                p = self.current_wp["pair"]
                self.visited.add(p); self.visited.add((p[1], p[0]))
                self.current_wp = None

        if new_wp is not None:
            if self.current_wp is None:
                self.current_wp = new_wp
            else:
                dist_to_curr = np.linalg.norm(self.current_wp["pos"] - self.boat_pos)
                if dist_to_curr > 80:
                    if new_wp["score"] > self.current_wp["score"] * wp_switch_thresh:
                        self.current_wp = new_wp

        if self.current_wp is not None:
            temp_visited = self.visited.copy()
            temp_visited.add(self.current_wp["pair"]); temp_visited.add((self.current_wp["pair"][1], self.current_wp["pair"][0]))
            vec = self.current_wp["pos"] - self.boat_pos
            next_head = math.atan2(vec[1], vec[0])
            self.next_wp = parameterized_find_gap(
                self.clusters, self.cluster_ids,
                self.current_wp["pos"], next_head,
                self.target, temp_visited,
                self.grid, self.dynamic_obstacles,
                align_exp, fwd_exp, clear_exp, width_exp, cluster_pen_w
            )
        else:
            self.next_wp = None

        self.path_timer += self.dt
        if self.path_timer >= 0.01:
            self.path_timer = 0
            goal = self.target if self.current_wp is None else self.current_wp["pos"]
            self.bezier_path = make_bezier_path(self.boat_pos, self.boat_heading, goal)
            if self.bezier_path is not None:
                self.pursuit_target = pure_pursuit(self.bezier_path, self.boat_pos, lookahead=52)
            if self.current_wp is not None and self.next_wp is not None:
                vec = self.current_wp["pos"] - self.boat_pos
                next_start_head = math.atan2(vec[1], vec[0])
                self.next_bezier_path = make_bezier_path(self.current_wp["pos"], next_start_head, self.next_wp["pos"])
                if self.next_bezier_path is not None:
                    self.next_pursuit_target = pure_pursuit(self.next_bezier_path, self.current_wp["pos"], lookahead=52)
            else:
                self.next_bezier_path = None
                self.next_pursuit_target = None

        if self.current_wp is not None and self.next_pursuit_target is not None and self.pursuit_target is not None:
            if np.linalg.norm(self.current_wp["pos"] - self.boat_pos) < 75:
                self.pursuit_target = self.next_pursuit_target

        center_idx = self.lidar_beams // 2
        span = self.lidar_beams // 9
        front_dists = dists[center_idx - span : center_idx + span]
        min_front_dist = np.min(front_dists)
        
        if min_front_dist < em_enter:
            self.emergency_mode = True
            self.emergency_cooldown = em_hold_frames
        elif self.emergency_mode:
            self.emergency_cooldown -= 1
            if min_front_dist > em_exit and self.emergency_cooldown <= 0:
                self.emergency_mode = False
        
        is_emergency = self.emergency_mode
        
        if self.pursuit_target is not None:
            px, py = self.pursuit_target
            heading_target = math.atan2(py - self.boat_pos[1], px - self.boat_pos[0])
            heading_error = wrap(heading_target - self.boat_heading)
            steer_raw = heading_error * steer_gain
            steer_f = steer_alpha * steer_raw + (1 - steer_alpha) * self.prev_steer
            self.prev_steer = steer_f
            
            avoid = reactive_avoidance(dists, self.rel_angles)
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
        target_fwd = (tL + tR) / (22.0 if is_emergency else 9.0)
        self.current_fwd = getattr(self, 'current_fwd', 0.0) * 0.95 + target_fwd * 0.05
        
        mom = (tR - tL) * mom_coeff
        hv = np.array([math.cos(self.boat_heading), math.sin(self.boat_heading)])
        acc = self.current_fwd / self.mass
        vel_norm = np.linalg.norm(self.boat_vel)
        drag = -self.drag * vel_norm * self.boat_vel if vel_norm > 0 else np.zeros(2)
        
        self.boat_vel += (acc * hv + drag) * self.dt
        self.boat_pos += self.boat_vel * self.dt
        
        ang_acc = (mom - self.rot_drag * self.boat_ang_vel) / self.inertia
        self.boat_ang_vel += ang_acc * self.dt
        self.boat_ang_vel *= 0.84
        self.boat_heading += self.boat_ang_vel * self.dt

        dist_to_target = np.linalg.norm(self.target - self.boat_pos)
        if self.collide(): return 'collide'
        if dist_to_target < 70: return 'goal'
        if self.frame > 2600: return 'timeout'
        return 'running'

_worker_sim = None

def worker_init():
    global _worker_sim
    _worker_sim = FastBoatSim()

def run_sim_task(args):
    global _worker_sim
    seed, params = args
    sim = _worker_sim
    sim.reset(seed)
    # 1100프레임으로 제한하여 불필요한 타임아웃 낭비 제거 및 4배 가속
    for _ in range(1100):
        status = sim.step_sim(params)
        if status != 'running':
            return status, sim.frame
    return 'timeout', 1100

def mutate_params(base, scale=0.03):
    p = copy.deepcopy(base)
    # 한 번에 모든 파라미터를 흔들지 않고 2~3개만 미세 조정하여 밸런스 붕괴 방지
    keys = list(p.keys())
    selected_keys = random.sample(keys, k=random.randint(2, 4))
    
    for k in selected_keys:
        if k == 'steer_gain':
            p[k] = float(np.clip(p[k] + np.random.normal(0, 0.015), 0.70, 0.90))
        elif k == 'steer_alpha':
            p[k] = float(np.clip(p[k] + np.random.normal(0, 0.012), 0.30, 0.45))
        elif k == 'mom_coeff':
            p[k] = float(np.clip(p[k] + np.random.normal(0, 0.00015), 0.0060, 0.0075))
        elif k == 'pwm_rng':
            p[k] = float(np.clip(p[k] + np.random.normal(0, 3.0), 250, 290))
        elif k == 'avoid_normal':
            p[k] = float(np.clip(p[k] + np.random.normal(0, 0.001), 0.015, 0.025))
        elif k == 'avoid_em':
            p[k] = float(np.clip(p[k] + np.random.normal(0, 0.005), 0.08, 0.14))
        elif k == 'clear_margin':
            p[k] = float(np.clip(p[k] + np.random.normal(0, 0.2), 1.5, 3.5))
        elif k == 'em_enter':
            p[k] = float(np.clip(p[k] + np.random.normal(0, 2.0), 105.0, 125.0))
        elif k == 'em_exit':
            p[k] = float(np.clip(p[k] + np.random.normal(0, 3.0), 145.0, 175.0))
        elif k == 'em_hold_frames':
            p[k] = int(np.clip(p[k] + int(np.random.choice([-1, 1])), 14, 22))
        elif k == 'align_exp':
            p[k] = float(np.clip(p[k] + np.random.normal(0, 0.2), 4.0, 6.0))
        elif k == 'fwd_exp':
            p[k] = float(np.clip(p[k] + np.random.normal(0, 0.15), 2.5, 3.8))
        elif k == 'clear_exp':
            p[k] = float(np.clip(p[k] + np.random.normal(0, 0.1), 1.2, 2.0))
        elif k == 'width_exp':
            p[k] = float(np.clip(p[k] + np.random.normal(0, 0.03), 0.15, 0.30))
        elif k == 'cluster_pen_w':
            p[k] = float(np.clip(p[k] + np.random.normal(0, 0.05), 0.3, 0.8))
        elif k == 'wp_switch_thresh':
            p[k] = float(np.clip(p[k] + np.random.normal(0, 0.02), 1.10, 1.25))
    return p

def main():
    n_workers = min(24, os.cpu_count() or 4)
    print("==================================================================", flush=True)
    print(f"  KABOAT Parallel Cluster Training (24 Workers, Headless Mode)", flush=True)
    print("==================================================================", flush=True)
    print("  Target: Normal & Emergency steering + 5 Waypoint Weights", flush=True)
    print("  Status: Real-time generation progress printed below", flush=True)
    print("------------------------------------------------------------------", flush=True)

    pool = mp.Pool(processes=n_workers, initializer=worker_init)
    
    best_params = {
        'steer_gain': 0.7752,
        'steer_alpha': 0.3515,
        'mom_coeff': 0.00665,
        'pwm_rng': 270.36,
        'avoid_normal': 0.019,
        'avoid_em': 0.11,
        'clear_margin': 2.15,
        'em_enter': 115.0,
        'em_exit': 160.0,
        'em_hold_frames': 18,
        'align_exp': 5.0,
        'fwd_exp': 3.0,
        'clear_exp': 1.5,
        'width_exp': 0.2,
        'cluster_pen_w': 0.5,
        'wp_switch_thresh': 1.15
    }

    best_rate = 88.0
    generation = 0

    try:
        while True:
            generation += 1
            seed_offset = generation * 1000 + 300000
            candidate = best_params if generation == 1 else mutate_params(best_params, scale=0.04 + 0.03 * (generation % 3 == 0))

            t0 = time.time()
            # 24코어 맞춤 96회 (24 x 4) 1단계 탐색
            tasks_stage1 = [(seed_offset + i, candidate) for i in range(96)]
            results_stage1 = pool.map(run_sim_task, tasks_stage1, chunksize=2)
            t_elapsed1 = time.time() - t0

            goals1 = sum(1 for r, f in results_stage1 if r == 'goal')
            collisions1 = sum(1 for r, f in results_stage1 if r == 'collide')
            timeouts1 = sum(1 for r, f in results_stage1 if r == 'timeout')
            rate1 = goals1 / len(results_stage1) * 100.0

            bar_len = 20
            filled_len = int(bar_len * rate1 / 100)
            bar = '#' * filled_len + '-' * (bar_len - filled_len)
            
            tag = "[ELITE]" if rate1 >= 90.0 else ""
            print(f"Gen {generation:03d} | [{bar}] {rate1:5.1f}% | Goal: {goals1:2d}/96 | Col: {collisions1:2d} | Time: {t_elapsed1:.2f}s {tag}", flush=True)

            if rate1 >= 88.0 or generation == 1:
                t0_2 = time.time()
                # 24코어 맞춤 120회 (24 x 5) 2단계 정밀 검증
                tasks_stage2 = [(seed_offset + 500 + i, candidate) for i in range(120)]
                results_stage2 = pool.map(run_sim_task, tasks_stage2, chunksize=2)
                t_elapsed2 = time.time() - t0_2

                goals2 = sum(1 for r, f in results_stage2 if r == 'goal')
                collisions2 = sum(1 for r, f in results_stage2 if r == 'collide')
                timeouts2 = sum(1 for r, f in results_stage2 if r == 'timeout')
                rate2 = goals2 / len(results_stage2) * 100.0

                bar2 = '#' * int(bar_len * rate2 / 100) + '-' * (bar_len - int(bar_len * rate2 / 100))
                print(f"   ↳ [Stage 2 Precision] [{bar2}] {rate2:5.1f}% ({goals2:3d}/120) | Col: {collisions2:2d} | Time: {t_elapsed2:.2f}s", flush=True)

                if rate2 >= best_rate:
                    best_rate = rate2
                    best_params = copy.deepcopy(candidate)
                    print(f"   * BEST RATE UPDATED: {best_rate:.1f}%", flush=True)
                    print(f"     Params: {best_params}", flush=True)
                    with open("best_learned_params.json", "w") as f:
                        json.dump(best_params, f, indent=2)

                if goals2 == 120 or rate2 >= 95.0:
                    print("\n==================================================================", flush=True)
                    print(f"  TARGET SUCCESS RATE ({rate2:.1f}%) ACHIEVED! BEST PARAMS SAVED.", flush=True)
                    print("==================================================================", flush=True)
                    for k, v in best_params.items():
                        print(f"    - {k:18s}: {v}")
                    break

    finally:
        pool.close()
        pool.join()

if __name__ == "__main__":
    main()
