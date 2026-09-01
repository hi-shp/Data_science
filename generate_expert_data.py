import os
import multiprocessing as mp
import numpy as np
import math
import copy
import time
import random

from perception import lidar_hits_np, update_grid, extract_clusters_from_grid, match_clusters
from navigation import target_is_clear, reactive_avoidance
from utils import wrap, make_bezier_path, pure_pursuit
from config import WIDTH, HEIGHT, GRID, GRID_W, GRID_H

def parameterized_find_gap(clusters, ids, boat_pos, boat_heading, target_pos, visited, grid, obstacles, params):
    align_exp = params.get('align_exp', 5.0)
    fwd_exp = params.get('fwd_exp', 3.0)
    clear_exp = params.get('clear_exp', 1.5)
    width_exp = params.get('width_exp', 0.2)
    cluster_pen_w = params.get('cluster_pen_w', 0.5)

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

def run_data_collection_task(args):
    global _worker_sim
    seed, params = args
    sim = _worker_sim
    sim.reset(seed)
    
    steer_gain = params.get('steer_gain', 0.7752)
    steer_alpha = params.get('steer_alpha', 0.3515)
    mom_coeff = params.get('mom_coeff', 0.00665)
    pwm_rng = params.get('pwm_rng', 270.36)
    avoid_normal = params.get('avoid_normal', 0.019)
    avoid_em = params.get('avoid_em', 0.11)
    clear_margin = params.get('clear_margin', 2.15)
    em_enter = params.get('em_enter', 78.0)
    em_exit = params.get('em_exit', 118.0)
    em_hold_frames = params.get('em_hold_frames', 14)
    wp_switch_thresh = params.get('wp_switch_thresh', 1.15)
    
    wp_arrive = 25.0
    max_frames = 2600

    states = []
    actions = []

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
                params
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
                params
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
                sim.pursuit_target = pure_pursuit(sim.bezier_path, sim.boat_pos, lookahead=70)
            if sim.current_wp is not None and sim.next_wp is not None:
                vec = sim.current_wp["pos"] - sim.boat_pos
                next_start_head = math.atan2(vec[1], vec[0])
                sim.next_bezier_path = make_bezier_path(sim.current_wp["pos"], next_start_head, sim.next_wp["pos"])
                if sim.next_bezier_path is not None:
                    sim.next_pursuit_target = pure_pursuit(sim.next_bezier_path, sim.current_wp["pos"], lookahead=70)
            else:
                sim.next_bezier_path = None
                sim.next_pursuit_target = None

        visual_target = sim.pursuit_target
        if sim.current_wp is not None and sim.next_pursuit_target is not None and sim.pursuit_target is not None:
            dist_to_wp = np.linalg.norm(sim.current_wp["pos"] - sim.boat_pos)
            if dist_to_wp < 75:
                sim.pursuit_target = sim.next_pursuit_target

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
            target_fwd = (tL + tR) / 22.0
            
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
        target_angle = math.atan2(sim.target[1] - sim.boat_pos[1], sim.target[0] - sim.boat_pos[0])
        rel_target_angle = wrap(target_angle - sim.boat_heading)
        
        lidar_norm = dists / sim.lidar_range
        target_dist_norm = np.clip(dist_to_target / 500.0, 0.0, 1.0)
        target_angle_norm = rel_target_angle / np.pi
        vel_norm_feat = np.clip(vel_norm / 100.0, 0.0, 1.0)
        ang_vel_norm = np.clip(sim.boat_ang_vel / 5.0, -1.0, 1.0)
        
        state = np.concatenate([
            lidar_norm, 
            [target_dist_norm, target_angle_norm, vel_norm_feat, ang_vel_norm]
        ]).astype(np.float32)
        
        out_L = (L - 1500) / 400.0
        out_R = (R - 1500) / 400.0
        action = np.array([out_L, out_R], dtype=np.float32)
        
        states.append(state)
        actions.append(action)

        if sim.collide():
            return False, None, None
        if dist_to_target < 70:
            return True, np.array(states), np.array(actions)

    return False, None, None

def main():
    import json
    try:
        with open("best_learned_params.json", "r") as f:
            params = json.load(f)
    except:
        params = {
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
        
    num_episodes = 2000
    n_workers = min(20, os.cpu_count() or 4)
    print(f"Generating expert data using {n_workers} workers...")
    
    pool = mp.Pool(processes=n_workers, initializer=worker_init)
    tasks = [(i, params) for i in range(num_episodes)]
    results = pool.imap_unordered(run_data_collection_task, tasks)
    
    all_states = []
    all_actions = []
    
    success_count = 0
    total_frames = 0
    
    for success, S, A in results:
        if success:
            success_count += 1
            all_states.append(S)
            all_actions.append(A)
            total_frames += len(S)
            if success_count % 50 == 0:
                print(f"Collected {success_count} successful episodes... ({total_frames} frames)")
                
    pool.close()
    pool.join()
    
    print(f"Finished. Total successful episodes: {success_count} / {num_episodes}")
    print(f"Total collected frames (Dataset Size): {total_frames}")
    
    final_S = np.concatenate(all_states, axis=0)
    final_A = np.concatenate(all_actions, axis=0)
    
    os.makedirs("data", exist_ok=True)
    np.savez_compressed("data/expert_data.npz", states=final_S, actions=final_A)
    print("Saved dataset to data/expert_data.npz")

if __name__ == "__main__":
    main()
