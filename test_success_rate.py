import os
import sys
import time
import math
import datetime
import pygame
import numpy as np

from environment import BoatEnv
from perception import lidar_hits_np, update_grid, extract_clusters_from_grid, match_clusters
from navigation import find_gap, target_is_clear, is_direct_target_safe
from utils import wrap, make_bezier_path, pure_pursuit

def main():
    env = BoatEnv()
    env.sim_speed = 4
    
    total_episodes = 10000
    completed_episodes = 0
    success_count = 0
    collision_count = 0
    
    start_time = time.time()
    out_file = "success_rate_1000.txt"
    
    def save_report(is_final=False):
        elapsed = time.time() - start_time
        sr = (success_count / completed_episodes * 100.0) if completed_episodes > 0 else 0.0
        cr = (collision_count / completed_episodes * 100.0) if completed_episodes > 0 else 0.0
        
        eta_seconds = (elapsed / completed_episodes) * (total_episodes - completed_episodes) if completed_episodes > 0 else 0
        eta_str = f"{eta_seconds/3600.0:.2f} hours ({eta_seconds/60.0:.1f} min)" if not is_final else "COMPLETED"
        
        status_str = "COMPLETED" if is_final else f"IN PROGRESS ({completed_episodes}/{total_episodes})"
        report = f"""============================================================
              SUCCESS RATE EVALUATION REPORT
============================================================
Status:          {status_str}
Last Updated:    {time.strftime('%Y-%m-%d %H:%M:%S')}
Total Episodes:  {total_episodes:,}
Completed:       {completed_episodes:,} / {total_episodes:,} ({completed_episodes/total_episodes*100:.2f}%)

Success Count:   {success_count:,}
Collision Count: {collision_count:,}

------------------------------------------------------------
>> SUCCESS RATE:   {sr:.2f} %
>> COLLISION RATE: {cr:.2f} %
------------------------------------------------------------
Elapsed Time:    {elapsed/3600.0:.2f} hours ({elapsed/60.0:.1f} min)
Estimated ETA:   {eta_str}
============================================================
"""
        with open(out_file, "w") as f:
            f.write(report)
            f.flush()
            os.fsync(f.fileno())
        print(f"[{completed_episodes:5d}/{total_episodes}] Success: {success_count} ({sr:5.2f}%) | Collision: {collision_count} ({cr:5.2f}%) | Elapsed: {elapsed/60.0:.1f}m | ETA: {eta_str}", flush=True)

    # 초기 파일 생성
    save_report(is_final=False)

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
                break
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:
                    env.handle_click(e.pos)

        if completed_episodes >= total_episodes:
            env.clock.tick(30)
            continue

        # 4배속 서브스텝 고속 반복 실행
        sub_steps = max(1, int(getattr(env, 'sim_speed', 4)))
        hits = None
        
        for _ in range(sub_steps):
            env.frame += 1
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
            boat_spd = math.hypot(env.boat_vel[0], env.boat_vel[1])
            clear_to_target = is_direct_target_safe(env.boat_pos, env.boat_heading, env.target, env.dynamic_obstacles, env.boat_radius, boat_spd)

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
                if target_is_clear(env.boat_pos, env.target, env.dynamic_obstacles):
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
                env.path_surf.fill((0, 0, 0, 0))
                boat_spd = math.hypot(env.boat_vel[0], env.boat_vel[1])
                if env.current_wp is None:
                    # 목적지 직행 상황에서도 회전 궤적 주변 장애물을 넉넉히 우회할 수 있도록 obstacles 전달
                    goal = env.target
                    env.bezier_path = make_bezier_path(env.boat_pos, env.boat_heading, goal, obstacles=env.dynamic_obstacles, boat_radius=env.boat_radius, boat_speed=boat_spd)
                else:
                    # 웨이포인트(갭) 우회 통과 구간: 속도 기반 선행 회전 및 장애물 외측 굴곡 곡률 부여
                    goal = env.current_wp["pos"]
                    env.bezier_path = make_bezier_path(env.boat_pos, env.boat_heading, goal, obstacles=env.dynamic_obstacles, boat_radius=env.boat_radius, boat_speed=boat_spd)
                    
                if env.bezier_path is not None:
                    env.pursuit_target = pure_pursuit(env.bezier_path, env.boat_pos, lookahead=65)
                    
                if env.current_wp is not None and env.next_wp is not None:
                    vec = env.current_wp["pos"] - env.boat_pos
                    next_head = math.atan2(vec[1], vec[0])
                    env.next_bezier_path = make_bezier_path(env.current_wp["pos"], next_head, env.next_wp["pos"], obstacles=env.dynamic_obstacles, boat_radius=env.boat_radius, boat_speed=boat_spd)
                    if env.next_bezier_path is not None:
                        env.next_pursuit_target = pure_pursuit(env.next_bezier_path, env.current_wp["pos"], lookahead=75)
                else:
                    env.next_bezier_path = None
                    env.next_pursuit_target = None

            visual_target = env.pursuit_target
            if env.current_wp is not None and env.next_pursuit_target is not None and env.pursuit_target is not None:
                dist_to_wp = np.linalg.norm(env.current_wp["pos"] - env.boat_pos)
                if dist_to_wp < 50:
                    env.pursuit_target = env.next_pursuit_target

            steer = env.update_steering(dists)
            if steer is None:
                steer = 0

            env.pursuit_target = visual_target
            L, R = env.get_pwm(steer)
            env.step(L, R)

            env.validate_wp_grid()
            env.validate_wp_obstacle_5x5()

            # 에피소드 종료 판정
            if env.collide() or np.linalg.norm(env.target - env.boat_pos) < 70:
                is_success = (np.linalg.norm(env.target - env.boat_pos) < 70 and not env.collide())
                if is_success:
                    success_count += 1
                else:
                    collision_count += 1
                completed_episodes += 1
                
                # 결과 파일 실시간 업데이트
                save_report(is_final=(completed_episodes >= total_episodes))
                
                env.reset()
                break

        if hits is not None:
            env.render(hits)
        env.clock.tick(0)  # 인위적 지연 없이 최대 속도로 시뮬레이션 가속

    pygame.quit()

if __name__ == '__main__':
    main()
