import pygame
import numpy as np
import math
import datetime
import os
from environment import BoatEnv
from perception import lidar_hits_np, update_grid, extract_clusters_from_grid, match_clusters
from navigation import find_gap, target_is_clear, is_direct_target_safe, is_waypoint_switch_safe, is_front_blocked
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

        # 실시간 배속 설정에 따른 서브스텝 반복 실행
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
            boat_spd = math.hypot(env.boat_vel[0], env.boat_vel[1])
            clear_to_target = is_direct_target_safe(env.boat_pos, env.boat_heading, env.target, env.dynamic_obstacles, env.boat_radius, boat_spd)

            # 목적지까지 회전 궤적 및 직선 경로에 장애물이 전혀 없을 때만 목적지 직행
            if clear_to_target:
                new_wp = None
                env.current_wp = None
                env.next_wp = None
                env.candidate_wps = []
            else:
                # 경로 상에 장애물이 있으면 장애물 사이 갭(웨이포인트)을 찾아 안전하게 우회
                new_wp = find_gap(
                    env.clusters, env.cluster_ids,
                    env.boat_pos, env.boat_heading,
                    env.target, env.visited,
                    env.grid, env.dynamic_obstacles,
                    params=env.params
                )
                if new_wp is not None:
                    env.candidate_wps = new_wp.get("candidates", [])
                elif env.current_wp is not None:
                    env.candidate_wps = env.current_wp.get("candidates", [])
                else:
                    env.candidate_wps = []

            if env.current_wp is not None:
                should_clear = False
                vec_to_wp = env.current_wp["pos"] - env.boat_pos
                dnow = np.linalg.norm(vec_to_wp)
                
                # 1. 웨이포인트 근접 시 해제 (25px 이내)
                if dnow < 25:
                    should_clear = True
                    
                wp_angle = math.atan2(vec_to_wp[1], vec_to_wp[0])
                angle_diff = abs(wrap(wp_angle - env.boat_heading))
                
                # 2. 웨이포인트를 지나쳐 측후방으로 넘어가면 즉시 해제하여 직진
                if angle_diff > np.pi / 2:
                    should_clear = True
                    
                # 3. 선박 위치에서 목적지까지 장애물이 없으면 즉시 해제하여 목적지 직행
                if target_is_clear(env.boat_pos, env.target, env.dynamic_obstacles):
                    should_clear = True
                    
                if should_clear:
                    p = env.current_wp["pair"]
                    env.visited.add(p)
                    env.visited.add((p[1], p[0]))
                    env.current_wp = None
                    env.candidate_wps = []

            # 1차 웨이포인트 양쪽 장애물의 실시간 위치 및 점수 갱신
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
                    # 동일한 웨이포인트가 계속 감지되면 최신 점수 및 가중치 분포 실시간 갱신
                    if new_wp is not None and (new_wp["pair"] == env.current_wp["pair"] or new_wp["pair"] == (id2, id1)):
                        env.current_wp["score"] = new_wp["score"]
                        if "factors" in new_wp:
                            env.current_wp["factors"] = new_wp["factors"]
                elif new_wp is not None:
                    # 기존 웨이포인트 부표가 시야에서 사라진 경우에만 새 웨이포인트로 안전하게 인계
                    if is_waypoint_switch_safe(env.boat_pos, env.boat_heading, env.current_wp["pos"], new_wp["pos"], env.dynamic_obstacles, env.boat_radius, boat_spd):
                        env.current_wp = new_wp

            if new_wp is not None:
                if env.current_wp is None:
                    env.current_wp = new_wp
                elif new_wp["pair"] != env.current_wp["pair"] and new_wp["pair"] != (env.current_wp["pair"][1], env.current_wp["pair"][0]):
                    # 다른 새로운 웨이포인트(갭)로 교체하려는 경우
                    dist_to_curr = np.linalg.norm(env.current_wp["pos"] - env.boat_pos)
                    # 전방 장애물 안전 거리 검사
                    front_blocked = is_front_blocked(env.boat_pos, env.boat_heading, env.dynamic_obstacles, env.boat_radius, block_dist=120.0, fov_deg=65.0)
                    if not front_blocked and dist_to_curr > 80:
                        # params에 설정된 wp_switch_thresh 실시간 적용 (기본: 1.15)
                        threshold = float(env.params.get('wp_switch_thresh', 1.15))
                        if new_wp["score"] > env.current_wp["score"] * threshold:
                            # 새 웨이포인트로 선회하는 부채꼴 및 베지어 궤적 상에 정면 장애물이 없을 때만 안전하게 스위칭
                            if is_waypoint_switch_safe(env.boat_pos, env.boat_heading, env.current_wp["pos"], new_wp["pos"], env.dynamic_obstacles, env.boat_radius, boat_spd):
                                env.current_wp = new_wp
                            
            if env.current_wp is not None and not clear_to_target:
                temp_visited = env.visited.copy()
                temp_visited.add(env.current_wp["pair"])
                temp_visited.add((env.current_wp["pair"][1], env.current_wp["pair"][0]))
                
                vec = env.current_wp["pos"] - env.boat_pos
                next_head = math.atan2(vec[1], vec[0])
                
                # 2차 갭 탐색 (1차 웨이포인트 이후 전방에 장애물 갭이 존재하면 주황색 2차 웨이포인트로 표출)
                env.next_wp = find_gap(
                    env.clusters, env.cluster_ids,
                    env.current_wp["pos"], next_head,
                    env.target, temp_visited,
                    env.grid, env.dynamic_obstacles,
                    params=env.params
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
                    env.pursuit_target = pure_pursuit(env.bezier_path, env.boat_pos, lookahead=70)
                    
                if env.current_wp is not None and env.next_wp is not None:
                    vec = env.current_wp["pos"] - env.boat_pos
                    next_start_head = math.atan2(vec[1], vec[0])
                    env.next_bezier_path = make_bezier_path(env.current_wp["pos"], next_start_head, env.next_wp["pos"], obstacles=env.dynamic_obstacles, boat_radius=env.boat_radius, boat_speed=boat_spd)
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

            if env.collide() or np.linalg.norm(env.target - env.boat_pos) < 70:
                is_success = (np.linalg.norm(env.target - env.boat_pos) < 70 and not env.collide())
                tag = "SUCCESS" if is_success else "FAIL"
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                outdir = r"screenshot"
                if not os.path.exists(outdir):
                    try:
                        os.makedirs(outdir)
                    except:
                        pass
                p = os.path.join(outdir, f"{ts}_{tag}.png")
                try:
                    if hits is not None:
                        env.render(hits)
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