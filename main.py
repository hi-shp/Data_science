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
        env.frame += 1
        
        env.update_dynamic_obstacles()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                return
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:
                    env.handle_click(e.pos)

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

        if target_is_clear(env.boat_pos, env.target, env.dynamic_obstacles):
            env.current_wp = None
            env.next_wp = None
            new_wp = None
        else:
            new_wp = find_gap(
                env.clusters, env.cluster_ids,
                env.boat_pos, env.boat_heading,
                env.target, env.visited,
                env.grid, env.dynamic_obstacles
            )

        # 정면 장애물 발생 시 중간 가상 우회 웨이포인트(Virtual Waypoint) 삽입
        center_idx = env.lidar_beams // 2
        span = env.lidar_beams // 10
        front_dists = dists[center_idx - span : center_idx + span]
        min_f_idx = int(np.argmin(front_dists))
        min_f_dist = front_dists[min_f_idx]
        
        if min_f_dist < 110.0 and env.current_wp is not None and not env.current_wp.get("is_virtual", False):
            obs_rel_ang = env.rel_angles[center_idx - span + min_f_idx]
            obs_glob_ang = env.boat_heading + obs_rel_ang
            obs_p = env.boat_pos + np.array([math.cos(obs_glob_ang), math.sin(obs_glob_ang)]) * min_f_dist
            
            perp_ang = obs_glob_ang + np.pi / 2.0
            normal = np.array([math.cos(perp_ang), math.sin(perp_ang)])
            
            d_left = obs_p - normal * 65.0
            d_right = obs_p + normal * 65.0
            
            l_space = np.sum(dists[center_idx - span*2 : center_idx])
            r_space = np.sum(dists[center_idx : center_idx + span*2])
            
            chosen_detour = d_left if l_space >= r_space else d_right
            
            env.next_wp = env.current_wp
            env.current_wp = {
                "pos": chosen_detour,
                "pair": (-1, -1),
                "score": 999.0,
                "is_virtual": True
            }

        if env.current_wp is not None:
            should_clear = False
            vec_to_wp = env.current_wp["pos"] - env.boat_pos
            dnow = np.linalg.norm(vec_to_wp)
            
            arrive_dist = 35.0 if env.current_wp.get("is_virtual", False) else 25.0
            if dnow < arrive_dist:
                should_clear = True
                
            wp_angle = math.atan2(vec_to_wp[1], vec_to_wp[0])
            angle_diff = abs(wrap(wp_angle - env.boat_heading))
            
            if angle_diff > np.pi / 2:
                should_clear = True
                
            if should_clear:
                p = env.current_wp["pair"]
                if p != (-1, -1):
                    env.visited.add(p)
                    env.visited.add((p[1], p[0]))
                env.current_wp = env.next_wp
                env.next_wp = None

        if new_wp is not None and (env.current_wp is None or not env.current_wp.get("is_virtual", False)):
            if env.current_wp is None:
                env.current_wp = new_wp
            else:
                dist_to_curr = np.linalg.norm(env.current_wp["pos"] - env.boat_pos)
                if dist_to_curr > 80:
                    threshold = 1.1
                    if new_wp["score"] > env.current_wp["score"] * threshold:
                        env.current_wp = new_wp
                        
        if env.current_wp is not None and not env.current_wp.get("is_virtual", False):
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
        elif env.current_wp is None:
            env.next_wp = None

        env.path_timer += env.dt
        if env.path_timer >= 0.01:
            env.path_timer = 0
            env.path_surf.fill((0, 0, 0, 0))
            
            if env.current_wp is None:
                goal = env.target
            else:
                goal = env.current_wp["pos"]
                
            env.bezier_path = make_bezier_path(env.boat_pos, env.boat_heading, goal)
            if env.bezier_path is not None:
                env.pursuit_target = pure_pursuit(env.bezier_path, env.boat_pos, lookahead=52)
                
            if env.current_wp is not None and env.next_wp is not None:
                vec = env.current_wp["pos"] - env.boat_pos
                next_start_head = math.atan2(vec[1], vec[0])
                env.next_bezier_path = make_bezier_path(env.current_wp["pos"], next_start_head, env.next_wp["pos"])
                if env.next_bezier_path is not None:
                    env.next_pursuit_target = pure_pursuit(env.next_bezier_path, env.current_wp["pos"], lookahead=52)
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

        env.render(hits)
        env.clock.tick(60)

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

if __name__ == "__main__":
    run()