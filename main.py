import pygame
import numpy as np
import math
import datetime
import os
from environment import BoatEnv
from perception import lidar_hits_np, update_grid, extract_clusters_from_grid, match_clusters
from navigation import find_gap, target_is_clear
from utils import wrap, make_bezier_path, pure_pursuit

def run():
    env = BoatEnv()

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                return
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:
                    env.handle_click(e.pos)

        # 실시간 배속 설정에 따른 서브스텝 반복 실행 (1x, 2x, 3x, 4x 실시간 즉시 반영)
        sub_steps = max(1, int(getattr(env, 'sim_speed', 1)))
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
            clear_to_target = target_is_clear(env.boat_pos, env.target, env.dynamic_obstacles)

            # 목적지까지 장애물이 없으면 거리와 무관하게 즉시 목적지 직행 (기존/신규 웨이포인트 모두 해제)
            if clear_to_target:
                new_wp = None
                env.current_wp = None
                env.next_wp = None
            else:
                # 경로 상에 장애물이 있으면 장애물 사이 갭(웨이포인트)을 찾아 안전하게 우회
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

            # 1차 웨이포인트 양쪽 장애물의 실시간 위치 추종 갱신
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
                
                if env.current_wp is None:
                    # 목적지까지 장애물이 없어 바로 직행하는 경우: 최소 곡률로 목적지 직진
                    goal = env.target
                    env.bezier_path = make_bezier_path(env.boat_pos, env.boat_heading, goal, obstacles=None)
                else:
                    # 웨이포인트(갭) 우회 통과 구간: 장애물을 넉넉히 둘러가도록 외측 굴곡 및 곡률 부여
                    goal = env.current_wp["pos"]
                    env.bezier_path = make_bezier_path(env.boat_pos, env.boat_heading, goal, obstacles=env.dynamic_obstacles, boat_radius=env.boat_radius)
                    
                if env.bezier_path is not None:
                    env.pursuit_target = pure_pursuit(env.bezier_path, env.boat_pos, lookahead=70)
                    
                if env.current_wp is not None and env.next_wp is not None:
                    vec = env.current_wp["pos"] - env.boat_pos
                    next_start_head = math.atan2(vec[1], vec[0])
                    env.next_bezier_path = make_bezier_path(env.current_wp["pos"], next_start_head, env.next_wp["pos"], obstacles=env.dynamic_obstacles, boat_radius=env.boat_radius)
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

            if env.collide() or np.linalg.norm(env.target - env.boat_pos) < 70:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                outdir = r"screenshot"
                if not os.path.exists(outdir):
                    try:
                        os.makedirs(outdir)
                    except:
                        pass
                p = os.path.join(outdir, f"{ts}.png")
                try:
                    pygame.image.save(env.screen, p)
                except:
                    pass
                env.reset()
                break

        if hits is not None:
            env.render(hits)
        env.clock.tick(60)

if __name__ == "__main__":
    run()