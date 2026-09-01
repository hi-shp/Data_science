import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame
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
    for _ in range(2600):
        status = sim.step_sim(params)
        if status != 'running':
            return status, sim.frame
    return 'timeout', 2600

def mutate_params(base, scale=0.05):
    p = copy.deepcopy(base)
    p['steer_gain'] = float(np.clip(p['steer_gain'] + np.random.normal(0, 0.02 * scale * 10), 0.70, 0.95))
    p['steer_alpha'] = float(np.clip(p['steer_alpha'] + np.random.normal(0, 0.015 * scale * 10), 0.30, 0.45))
    p['mom_coeff'] = float(np.clip(p['mom_coeff'] + np.random.normal(0, 0.0002 * scale * 10), 0.0060, 0.0078))
    p['pwm_rng'] = float(np.clip(p['pwm_rng'] + np.random.normal(0, 5 * scale * 10), 250, 300))
    p['avoid_normal'] = float(np.clip(p['avoid_normal'] + np.random.normal(0, 0.0015 * scale * 10), 0.015, 0.030))
    p['avoid_em'] = float(np.clip(p['avoid_em'] + np.random.normal(0, 0.008 * scale * 10), 0.07, 0.16))
    p['clear_margin'] = float(np.clip(p['clear_margin'] + np.random.normal(0, 0.4 * scale * 10), 1.5, 4.0))
    
    p['em_enter'] = float(np.clip(p['em_enter'] + np.random.normal(0, 3.0 * scale * 10), 95.0, 135.0))
    p['em_exit'] = float(np.clip(p['em_exit'] + np.random.normal(0, 4.0 * scale * 10), 135.0, 185.0))
    p['em_hold_frames'] = int(np.clip(p['em_hold_frames'] + int(np.random.normal(0, 2 * scale * 10)), 12, 25))
    
    p['align_exp'] = float(np.clip(p['align_exp'] + np.random.normal(0, 0.3 * scale * 10), 3.0, 7.0))
    p['fwd_exp'] = float(np.clip(p['fwd_exp'] + np.random.normal(0, 0.2 * scale * 10), 2.0, 4.5))
    p['clear_exp'] = float(np.clip(p['clear_exp'] + np.random.normal(0, 0.15 * scale * 10), 1.0, 2.5))
    p['width_exp'] = float(np.clip(p['width_exp'] + np.random.normal(0, 0.05 * scale * 10), 0.1, 0.4))
    p['wp_switch_thresh'] = float(np.clip(p['wp_switch_thresh'] + np.random.normal(0, 0.03 * scale * 10), 1.08, 1.35))
    return p

def draw_mini_sim(surf, sim, rect, worker_id):
    rx, ry, rw, rh = rect
    sub = pygame.Surface((rw, rh))
    sub.fill((10, 16, 26))
    
    sx = rw / sim.w
    sy = rh / sim.sim_h
    
    # 장애물 그리기
    for (ox, oy, r) in sim.dynamic_obstacles:
        pygame.draw.circle(sub, (40, 80, 110), (int(ox * sx), int(oy * sy)), max(2, int(r * sx)))
        
    # 목표점
    pygame.draw.circle(sub, (255, 200, 50), (int(sim.target[0] * sx), int(sim.target[1] * sy)), 5)
    
    # 경로
    if sim.bezier_path is not None and len(sim.bezier_path) > 1:
        pts = [(int(p[0] * sx), int(p[1] * sy)) for p in sim.bezier_path]
        pygame.draw.lines(sub, (0, 220, 200), False, pts, 2)
        
    # 보트
    bx, by = int(sim.boat_pos[0] * sx), int(sim.boat_pos[1] * sy)
    color = (255, 60, 80) if sim.emergency_mode else (80, 220, 120)
    pygame.draw.circle(sub, color, (bx, by), 4)
    hx = bx + int(math.cos(sim.boat_heading) * 10)
    hy = by + int(math.sin(sim.boat_heading) * 10)
    pygame.draw.line(sub, (255, 255, 255), (bx, by), (hx, hy), 2)
    
    # 테두리 및 라벨
    pygame.draw.rect(sub, (30, 55, 80), (0, 0, rw, rh), 1)
    surf.blit(sub, (rx, ry))

def main():
    n_workers = min(24, os.cpu_count() or 4)
    pygame.init()
    pygame.font.init()
    
    W_SCREEN, H_SCREEN = 1260, 720
    screen = pygame.display.set_mode((W_SCREEN, H_SCREEN))
    pygame.display.set_caption(f"KABOAT Parallel Cluster Training ({n_workers} Workers)")
    clock = pygame.time.Clock()
    
    font_lg = pygame.font.SysFont("notosans", 22, bold=True)
    font_md = pygame.font.SysFont("notosans", 16, bold=True)
    font_sm = pygame.font.SysFont("notosans", 13)
    
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
    history_rates = []
    
    # 4개의 실시간 시각화용 시뮬레이터 인스턴스
    vis_sims = [FastBoatSim() for _ in range(4)]
    for idx, s in enumerate(vis_sims):
        s.reset(seed=1000 + idx)

    print(f"Cluster RL training started with {n_workers} CPU workers.", flush=True)

    running = True
    try:
        while running:
            generation += 1
            seed_offset = generation * 1000 + 200000
            candidate = best_params if generation == 1 else mutate_params(best_params, scale=0.04 + 0.03 * (generation % 3 == 0))

            # 4개의 시각화 시뮬레이터 리셋
            for idx, s in enumerate(vis_sims):
                s.reset(seed=seed_offset + idx)

            # 비동기 병렬 평가 시작 (24코어 백그라운드)
            tasks_stage1 = [(seed_offset + i, candidate) for i in range(48)]
            async_res1 = pool.map_async(run_sim_task, tasks_stage1, chunksize=2)

            t0 = time.time()
            
            # 백그라운드 24코어가 돌 동안 메인 화면에서 4개 보트 실시간 렌더링
            while not async_res1.ready() and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                        
                # 4개 시뮬레이터 1스텝 전진
                for s in vis_sims:
                    st = s.step_sim(candidate)
                    if st != 'running':
                        s.reset(seed=random.randint(10000, 99999))

                # --- 렌더링 루틴 ---
                screen.fill((7, 11, 18))
                
                # 1. 2x2 멀티 뷰포트 (4개 화면)
                vw, vh = 410, 320
                coords = [(15, 60), (435, 60), (15, 385), (435, 385)]
                for idx, s in enumerate(vis_sims):
                    draw_mini_sim(screen, s, (coords[idx][0], coords[idx][1], vw, vh), idx + 1)
                    lbl = font_sm.render(f"Worker Monitor #{idx+1} [Live]", True, (130, 170, 200))
                    screen.blit(lbl, (coords[idx][0] + 8, coords[idx][1] + 8))

                # 2. 상단 헤더
                hdr_txt = font_lg.render(f"PARALLEL CLUSTER TRAINING - 24 CPU WORKERS", True, (0, 230, 190))
                gen_txt = font_md.render(f"GENERATION: {generation:03d}  |  STATUS: EVALUATING 48 BATCHES", True, (180, 210, 230))
                screen.blit(hdr_txt, (15, 12))
                screen.blit(gen_txt, (15, 35))

                # 3. 우측 대시보드 패널
                px = 860
                pygame.draw.rect(screen, (14, 22, 35), (px, 15, 385, 690), border_radius=8)
                pygame.draw.rect(screen, (30, 50, 75), (px, 15, 385, 690), 1, border_radius=8)

                title_hud = font_md.render("REAL-TIME RL METRICS", True, (255, 200, 50))
                screen.blit(title_hud, (px + 15, 25))

                best_txt = font_lg.render(f"BEST RATE: {best_rate:.1f}%", True, (0, 255, 140))
                screen.blit(best_txt, (px + 15, 55))

                # 그래프 영역
                graph_rect = pygame.Rect(px + 15, 95, 355, 110)
                pygame.draw.rect(screen, (8, 14, 22), graph_rect)
                pygame.draw.rect(screen, (25, 45, 65), graph_rect, 1)
                
                if len(history_rates) > 1:
                    pts = []
                    for i, r in enumerate(history_rates[-30:]):
                        gx = graph_rect.x + int(i / max(1, len(history_rates[-30:]) - 1) * graph_rect.w)
                        gy = graph_rect.bottom - int((r / 100.0) * graph_rect.h)
                        pts.append((gx, gy))
                    pygame.draw.lines(screen, (0, 220, 255), False, pts, 2)
                    
                rate_lbl = font_sm.render(f"History (Last 30 Gen) - Current Target: 95.0%", True, (120, 150, 180))
                screen.blit(rate_lbl, (px + 15, 212))

                # 파라미터 리스트 출력
                param_title = font_md.render("BEST OPTIMIZED PARAMETERS", True, (200, 220, 240))
                screen.blit(param_title, (px + 15, 245))
                
                y_off = 275
                items = [
                    ("Steer Gain", f"{best_params['steer_gain']:.4f}"),
                    ("Steer Alpha", f"{best_params['steer_alpha']:.4f}"),
                    ("Moment Coeff", f"{best_params['mom_coeff']:.6f}"),
                    ("PWM Range", f"{best_params['pwm_rng']:.1f}"),
                    ("Emergency Enter", f"{best_params['em_enter']:.1f} px"),
                    ("Emergency Exit", f"{best_params['em_exit']:.1f} px"),
                    ("Emergency Hold", f"{best_params['em_hold_frames']} frames"),
                    ("Gap Align Exp", f"{best_params['align_exp']:.2f}"),
                    ("Gap Fwd Exp", f"{best_params['fwd_exp']:.2f}"),
                    ("Clear Exp", f"{best_params['clear_exp']:.2f}"),
                    ("Width Exp", f"{best_params['width_exp']:.2f}"),
                    ("Cluster Penalty", f"{best_params['cluster_pen_w']:.2f}"),
                    ("WP Switch Thresh", f"{best_params['wp_switch_thresh']:.2f}"),
                ]
                
                for k, v in items:
                    t_k = font_sm.render(k, True, (130, 160, 185))
                    t_v = font_sm.render(v, True, (255, 255, 255))
                    screen.blit(t_k, (px + 15, y_off))
                    screen.blit(t_v, (px + 230, y_off))
                    y_off += 28

                pygame.display.flip()
                clock.tick(60)

            if not running: break

            results_stage1 = async_res1.get()
            t_elapsed1 = time.time() - t0

            goals1 = sum(1 for r, f in results_stage1 if r == 'goal')
            collisions1 = sum(1 for r, f in results_stage1 if r == 'collide')
            rate1 = goals1 / len(results_stage1) * 100.0
            history_rates.append(rate1)

            print(f"Gen {generation:03d} | Stage 1 (48) Goals: {goals1:2d} ({rate1:.1f}%) | Collisions: {collisions1:2d} | {t_elapsed1:.1f}s", flush=True)

            if rate1 >= 88.0 or generation == 1:
                tasks_stage2 = [(seed_offset + 500 + i, candidate) for i in range(100)]
                results_stage2 = pool.map(run_sim_task, tasks_stage2, chunksize=4)

                goals2 = sum(1 for r, f in results_stage2 if r == 'goal')
                collisions2 = sum(1 for r, f in results_stage2 if r == 'collide')
                rate2 = goals2 / 100.0 * 100.0

                print(f"   ↳ Stage 2 Precision (100) Goals: {goals2:3d}/100 ({rate2:.1f}%) | Collisions: {collisions2:2d}", flush=True)

                if rate2 >= best_rate:
                    best_rate = rate2
                    best_params = copy.deepcopy(candidate)
                    print(f"   ★ Best Rate Updated: {best_rate:.1f}%", flush=True)
                    with open("best_learned_params.json", "w") as f:
                        json.dump(best_params, f, indent=2)

                if goals2 == 100 or rate2 >= 95.0:
                    print("\n==================================================================", flush=True)
                    print(f"  Target Success Rate ({rate2:.1f}%) Achieved! Final Parameters Saved.", flush=True)
                    print("==================================================================", flush=True)
                    break

    finally:
        pool.close()
        pool.join()
        pygame.quit()

if __name__ == "__main__":
    main()
