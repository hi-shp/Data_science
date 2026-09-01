import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import time
import math
import random
import json
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import pygame
pygame.init()

from environment import BoatEnv
from perception import lidar_hits_np, update_grid, extract_clusters_from_grid, match_clusters
from navigation import target_is_clear, find_gap, reactive_avoidance
from utils import wrap, make_bezier_path, pure_pursuit

def run_single_episode(seed, max_frames=4000):
    random.seed(seed)
    np.random.seed(seed)
    
    # 헤드리스 환경 생성
    env = BoatEnv()
    
    # 시드 기반 재설정
    env.boat_pos = np.array([65, env.sim_h/2], dtype=np.float32)
    env.boat_vel = np.zeros(2, dtype=np.float32)
    env.boat_ang_vel = 0.0
    env.boat_heading = 0.0
    env.target = np.array([env.w - 100, env.sim_h/2], dtype=np.float32)
    
    # 장애물 생성
    obs = []
    t = 0
    while len(obs) < env.obs_n and t < 5000:
        t += 1
        x = random.randint(300, env.w - 300)
        y = random.randint(30, env.sim_h - 30)
        p = np.array([x, y])
        if np.linalg.norm(p - env.target) < 180: continue
        if np.linalg.norm(p - env.boat_pos) < 180: continue
        
        ok = True
        for (ox, oy, r) in obs:
            if np.linalg.norm(p - np.array([ox, oy])) < env.min_obs:
                ok = False
                break
        if ok:
            obs.append((x, y, env.obs_r))
            
    env.obstacles = np.array(obs, dtype=np.float32)
    env.dynamic_obstacles = env.obstacles.copy()
    
    status = 'RUNNING'
    for frame in range(1, max_frames + 1):
        env.frame = frame
        env.update_dynamic_obstacles()
        
        dists, hits = lidar_hits_np(
            env.boat_pos, env.boat_heading,
            env.rel_angles, env.dynamic_obstacles,
            env.lidar_range
        )
        
        update_grid(env.grid, hits)
        env.grid *= 0.945
        
        new_c = extract_clusters_from_grid(env.grid)
        env.clusters, env.cluster_ids = match_clusters(
            env.clusters, env.cluster_ids, new_c
        )
        
        dist_to_target = np.linalg.norm(env.target - env.boat_pos)
        clear_to_target = target_is_clear(env.boat_pos, env.target, env.dynamic_obstacles)
        
        if clear_to_target:
            new_wp = None
            env.current_wp = None
            env.next_wp = None
        else:
            new_wp = find_gap(
                env.clusters, env.cluster_ids,
                env.boat_pos, env.boat_heading,
                env.target, env.visited,
                env.grid, env.dynamic_obstacles
            )
            
        if env.current_wp is not None:
            should_clear = False
            vec_to_wp = env.current_wp["pos"] - env.boat_pos
            dnow = np.linalg.norm(vec_to_wp)
            if dnow < 25:
                should_clear = True
            wp_angle = math.atan2(vec_to_wp[1], vec_to_wp[0])
            angle_diff = abs(wrap(wp_angle - env.boat_heading))
            if angle_diff > np.pi / 2:
                should_clear = True
            if should_clear:
                p = env.current_wp["pair"]
                env.visited.add(p)
                env.visited.add((p[1], p[0]))
                env.current_wp = None
                
        if env.current_wp is not None:
            id1, id2 = env.current_wp["pair"]
            if id1 in env.cluster_ids and id2 in env.cluster_ids:
                idx1 = env.cluster_ids.index(id1)
                idx2 = env.cluster_ids.index(id2)
                c1_now = env.clusters[idx1]
                c2_now = env.clusters[idx2]
                env.current_wp["c1"] = c1_now
                env.current_wp["c2"] = c2_now
                env.current_wp["pos"] = (c1_now + c2_now) / 2.0
            elif new_wp is not None:
                env.current_wp = new_wp
                
        if new_wp is not None:
            if env.current_wp is None:
                env.current_wp = new_wp
            else:
                dist_to_curr = np.linalg.norm(env.current_wp["pos"] - env.boat_pos)
                if dist_to_curr > 80:
                    threshold = 1.1
                    if new_wp["score"] > env.current_wp["score"] * threshold:
                        env.current_wp = new_wp
                        
        if env.current_wp is not None and not clear_to_target:
            temp_visited = env.visited.copy()
            temp_visited.add(env.current_wp["pair"])
            temp_visited.add((env.current_wp["pair"][1], env.current_wp["pair"][0]))
            vec = env.current_wp["pos"] - env.boat_pos
            next_head = math.atan2(vec[1], vec[0])
            env.next_wp = find_gap(
                env.clusters, env.cluster_ids,
                env.current_wp["pos"], next_head,
                env.target, temp_visited,
                env.grid, env.dynamic_obstacles
            )
        else:
            env.next_wp = None
            
        env.path_timer += env.dt
        if env.path_timer >= 0.01:
            env.path_timer = 0
            if env.current_wp is None:
                goal = env.target
            else:
                goal = env.current_wp["pos"]
                
            env.bezier_path = make_bezier_path(env.boat_pos, env.boat_heading, goal, env.dynamic_obstacles, env.boat_radius)
            if env.bezier_path is not None:
                env.pursuit_target = pure_pursuit(env.bezier_path, env.boat_pos, lookahead=75)
                
            if env.current_wp is not None and env.next_wp is not None:
                vec = env.current_wp["pos"] - env.boat_pos
                next_start_head = math.atan2(vec[1], vec[0])
                env.next_bezier_path = make_bezier_path(env.current_wp["pos"], next_start_head, env.next_wp["pos"], env.dynamic_obstacles, env.boat_radius)
                if env.next_bezier_path is not None:
                    env.next_pursuit_target = pure_pursuit(env.next_bezier_path, env.current_wp["pos"], lookahead=75)
            else:
                env.next_bezier_path = None
                env.next_pursuit_target = None
                
        visual_target = env.pursuit_target
        if env.current_wp is not None and env.next_pursuit_target is not None and env.pursuit_target is not None:
            dist_to_wp = np.linalg.norm(env.current_wp["pos"] - env.boat_pos)
            if dist_to_wp < 85:
                env.pursuit_target = env.next_pursuit_target
                
        steer = env.update_steering(dists)
        if steer is None:
            steer = 0
            
        env.pursuit_target = visual_target
        L, R = env.get_pwm(steer)
        env.step(L, R)
        
        env.validate_wp_grid()
        env.validate_wp_obstacle_5x5()
        
        if env.collide():
            status = 'COLLISION'
            break
            
        if np.linalg.norm(env.target - env.boat_pos) < 70:
            status = 'SUCCESS'
            break
            
    if status == 'RUNNING':
        status = 'TIMEOUT'
        
    return {
        'seed': seed,
        'status': status,
        'frames': frame,
        'final_dist': float(np.linalg.norm(env.target - env.boat_pos))
    }

def main():
    total_episodes = 1000
    num_workers = max(1, mp.cpu_count() - 2)
    print(f"[*] 1,000회 시뮬레이션 평가 시작 (병렬 프로세스: {num_workers}개)")
    
    start_time = time.time()
    results = []
    success_count = 0
    collision_count = 0
    timeout_count = 0
    
    log_file = "evaluation_1000_results.log"
    summary_file = "evaluation_1000_summary.json"
    
    with open(log_file, "w") as f_log:
        f_log.write(f"Evaluation Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_log.write("Episode | Seed | Status | Frames | Final Distance\n")
        f_log.write("-" * 55 + "\n")
        
        seeds = [100000 + i * 37 for i in range(total_episodes)]
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {executor.submit(run_single_episode, s): i for i, s in enumerate(seeds)}
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                res = future.result()
                results.append(res)
                
                if res['status'] == 'SUCCESS':
                    success_count += 1
                elif res['status'] == 'COLLISION':
                    collision_count += 1
                else:
                    timeout_count += 1
                    
                done_count = len(results)
                sr = (success_count / done_count) * 100.0
                cr = (collision_count / done_count) * 100.0
                
                log_line = f"[{done_count:4d}/{total_episodes}] Seed: {res['seed']} | {res['status']:9s} | Frames: {res['frames']:4d} | Dist: {res['final_dist']:6.1f}px | Success Rate: {sr:5.1f}%"
                f_log.write(log_line + "\n")
                f_log.flush()
                
                if done_count % 25 == 0 or done_count == total_episodes:
                    elapsed = time.time() - start_time
                    fps = done_count / elapsed
                    print(f"진행: {done_count:4d}/{total_episodes} ({done_count/total_episodes*100:5.1f}%) | 성공: {success_count} ({sr:5.1f}%) | 충돌: {collision_count} ({cr:5.1f}%) | 경과: {elapsed:.1f}초 ({fps:.1f} ep/s)")
                    
                    summary_data = {
                        'total_episodes': total_episodes,
                        'completed': done_count,
                        'success_count': success_count,
                        'collision_count': collision_count,
                        'timeout_count': timeout_count,
                        'success_rate_pct': round(sr, 2),
                        'collision_rate_pct': round(cr, 2),
                        'timeout_rate_pct': round((timeout_count / done_count) * 100.0, 2),
                        'avg_frames': float(np.mean([r['frames'] for r in results])),
                        'elapsed_seconds': round(elapsed, 1),
                        'status': 'IN_PROGRESS' if done_count < total_episodes else 'COMPLETED',
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    with open(summary_file, "w") as f_sum:
                        json.dump(summary_data, f_sum, indent=2)

    total_time = time.time() - start_time
    final_sr = (success_count / total_episodes) * 100.0
    print("\n" + "=" * 60)
    print(f"[*] 1,000회 평가 완료! 총 소요 시간: {total_time:.1f}초")
    print(f"[*] 성공 횟수: {success_count} / {total_episodes} ({final_sr:.2f}%)")
    print(f"[*] 충돌 횟수: {collision_count} / {total_episodes} ({(collision_count/total_episodes)*100:.2f}%)")
    print(f"[*] 타임아웃: {timeout_count} / {total_episodes} ({(timeout_count/total_episodes)*100:.2f}%)")
    print("=" * 60)

if __name__ == '__main__':
    main()
