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
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    env.paused = not env.paused
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:
                    env.handle_click(e.pos)

        if getattr(env, 'paused', False):
            dists, hits = lidar_hits_np(
                env.boat_pos, env.boat_heading,
                env.rel_angles, env.dynamic_obstacles,
                env.lidar_range
            )
            env.render(hits)
            env.clock.tick(60)
            continue

        # 실시간 배속 설정에 따른 서브스텝 반복 실행
        sub_steps = max(1, int(getattr(env, 'sim_speed', 1)))
        plan_interval = 1 if sub_steps <= 2 else (2 if sub_steps <= 4 else 3)
        hits = None
        new_wp = None
        
        for step_idx in range(sub_steps):
            env.frame += 1
            env.update_dynamic_obstacles()

            dists, hits = lidar_hits_np(
                env.boat_pos, env.boat_heading,
                env.rel_angles, env.dynamic_obstacles,
                env.lidar_range
            )

            update_grid(env.grid, hits)
            env.grid *= 0.945

            # 연산 부하 절감을 위한 적응형 인지/탐색 주기
            should_plan = (step_idx % plan_interval == 0 or step_idx == sub_steps - 1)
            if should_plan:
                new_c = extract_clusters_from_grid(env.grid)
                env.clusters, env.cluster_ids = match_clusters(
                    env.clusters, env.cluster_ids, new_c
                )

                # [Gaps 버튼 전용 데이터] 전방 180도(헤딩 기준 좌우 ±90도: 전방 내적 >= 0) 장애물만 기준으로 모든 갭(N C 2) 생성
                bx, by = env.boat_pos
                ch = math.cos(env.boat_heading)
                sh = math.sin(env.boat_heading)
                front_clusters = [c for c in env.clusters if (c[0] - bx) * ch + (c[1] - by) * sh >= 0]

                gui_all_gaps = []
                n_fc = len(front_clusters)
                for i in range(n_fc):
                    c1 = front_clusters[i]
                    for j in range(i + 1, n_fc):
                        c2 = front_clusters[j]
                        mid_pt = (c1 + c2) / 2.0
                        gui_all_gaps.append({
                            "pos": mid_pt.copy(),
                            "c1": c1.copy(),
                            "c2": c2.copy()
                        })
                env.all_gaps = gui_all_gaps
                env.total_gaps_count = len(gui_all_gaps)

                dist_to_target = math.hypot(env.target[0] - env.boat_pos[0], env.target[1] - env.boat_pos[1])
                boat_spd = math.hypot(env.boat_vel[0], env.boat_vel[1])
                clear_to_target = is_direct_target_safe(env.boat_pos, env.boat_heading, env.target, env.dynamic_obstacles, env.boat_radius, boat_spd, params=env.params)

                # 목적지 방향 직선 시야(경로)에 장애물이 없으면 즉시 목적지 직행
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
            else:
                boat_spd = math.hypot(env.boat_vel[0], env.boat_vel[1])

            if env.current_wp is not None:
                should_clear = False
                c1 = env.current_wp.get("c1")
                c2 = env.current_wp.get("c2")
                mid = env.current_wp["pos"]
                vec_to_wp = mid - env.boat_pos
                dnow = math.hypot(vec_to_wp[0], vec_to_wp[1])
                
                # 1. 웨이포인트 중심점 근접 시 즉시 해제 (28px 이내)
                if dnow < 60:
                    should_clear = True
                    
                # 2. 웨이포인트 게이트 선 통과 판정 (c1, c2 사이 게이트 선을 전방으로 통과 시 즉시 해제)
                if not should_clear and c1 is not None and c2 is not None:
                    vgx = c2[0] - c1[0]; vgy = c2[1] - c1[1]
                    gate_len = math.hypot(vgx, vgy)
                    if gate_len > 1e-3:
                        ugx = vgx / gate_len; ugy = vgy / gate_len
                        ngx = -ugy; ngy = ugx
                        hx = math.cos(env.boat_heading); hy = math.sin(env.boat_heading)
                        if ngx * hx + ngy * hy < 0:
                            ngx = -ngx; ngy = -ngy
                        rbx = env.boat_pos[0] - mid[0]; rby = env.boat_pos[1] - mid[1]
                        d_normal = rbx * ngx + rby * ngy
                        d_lateral = abs(rbx * ugx + rby * ugy)
                        if 15.0 <= d_normal < 60.0 and d_lateral < (gate_len / 2.0 + 20.0):
                            should_clear = True
                            
                # 3. 웨이포인트를 이미 지나쳐 측후방으로 넘어간 경우 (95도 이상 & 75px 이내)
                if not should_clear:
                    wp_angle = math.atan2(vec_to_wp[1], vec_to_wp[0])
                    angle_diff = abs(wrap(wp_angle - env.boat_heading))
                    if angle_diff > 1.6580627893946132 and dnow < 75:  # np.deg2rad(95)
                        should_clear = True
                    
                if should_clear:
                    p = env.current_wp["pair"]
                    env.visited.add(p)
                    env.visited.add((p[1], p[0]))
                    # 1차 웨이포인트 통과 시 2차 웨이포인트가 미리 감지되어 있으면 부드럽게 1차로 승격
                    if env.next_wp is not None:
                        env.current_wp = env.next_wp
                        env.next_wp = None
                    else:
                        env.current_wp = None
                        env.total_gaps_count = 0
                        env.all_gaps = []
                    env.candidate_wps = []

            # 1차 웨이포인트 양쪽 장애물의 실시간 위치 및 점수 갱신
            if env.current_wp is not None:
                id1, id2 = env.current_wp["pair"]
                matched = False
                if id1 in env.cluster_ids and id2 in env.cluster_ids:
                    idx1 = env.cluster_ids.index(id1)
                    idx2 = env.cluster_ids.index(id2)
                    c1_now = env.clusters[idx1]
                    c2_now = env.clusters[idx2]
                    env.current_wp["c1"] = c1_now
                    env.current_wp["c2"] = c2_now
                    env.current_wp["pos"] = (c1_now + c2_now) / 2.0
                    matched = True
                    if new_wp is not None and (new_wp["pair"] == env.current_wp["pair"] or new_wp["pair"] == (id2, id1)):
                        env.current_wp["score"] = new_wp["score"]
                        if "factors" in new_wp:
                            env.current_wp["factors"] = new_wp["factors"]
                
                # ID가 변경되었더라도 기존 부표 물리 좌표(c1, c2)와 가까운 클러스터(35px 이내)로 안정적 추종
                if not matched:
                    c1_old = env.current_wp.get("c1")
                    c2_old = env.current_wp.get("c2")
                    if c1_old is not None and c2_old is not None and len(env.clusters) >= 2:
                        cl_arr = np.array(env.clusters)  # (N, 2)
                        d1 = np.sqrt(np.sum((cl_arr - c1_old)**2, axis=1))
                        d2 = np.sqrt(np.sum((cl_arr - c2_old)**2, axis=1))
                        i1, i2 = int(np.argmin(d1)), int(np.argmin(d2))
                        if d1[i1] < 35.0 and d2[i2] < 35.0 and i1 != i2:
                            env.current_wp["c1"] = env.clusters[i1]
                            env.current_wp["c2"] = env.clusters[i2]
                            env.current_wp["pos"] = (env.clusters[i1] + env.clusters[i2]) / 2.0
                            env.current_wp["pair"] = (env.cluster_ids[i1], env.cluster_ids[i2])

            # 1차 웨이포인트가 비어있을 때만 새로운 웨이포인트 최초 지정 (접근 중인 1차 WP를 전방 2차 WP로 덮어쓰지 않음)
            if new_wp is not None and env.current_wp is None:
                env.current_wp = new_wp
                            
            if should_plan:
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
                        params=env.params,
                        is_next_wp=True
                    )
                else:
                    env.next_wp = None

            env.path_timer += env.dt
            if env.path_timer >= 0.01:
                env.path_timer = 0
                
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
                    if env.bezier_path is not None and len(env.bezier_path) >= 2:
                        t1 = env.bezier_path[-1] - env.bezier_path[-2]
                        if np.linalg.norm(t1) > 1e-6:
                            next_start_head = math.atan2(t1[1], t1[0])
                        else:
                            vec = env.current_wp["pos"] - env.boat_pos
                            next_start_head = math.atan2(vec[1], vec[0])
                    else:
                        vec = env.current_wp["pos"] - env.boat_pos
                        next_start_head = math.atan2(vec[1], vec[0])

                    env.next_bezier_path = make_bezier_path(
                        env.current_wp["pos"], next_start_head, env.next_wp["pos"],
                        obstacles=env.dynamic_obstacles, boat_radius=env.boat_radius, boat_speed=boat_spd,
                        start_tangent_fixed=True
                    )
                    if env.next_bezier_path is not None:
                        env.next_pursuit_target = pure_pursuit(env.next_bezier_path, env.current_wp["pos"], lookahead=75)
                else:
                    env.next_bezier_path = None
                    env.next_pursuit_target = None

            steer = env.update_steering(dists)
            if steer is None:
                steer = 0


            L, R = env.get_pwm(steer)
            env.step(L, R)

            env.validate_wp_grid()
            env.validate_wp_obstacle_5x5()

            dist_tgt_end = math.hypot(env.target[0] - env.boat_pos[0], env.target[1] - env.boat_pos[1])
            if env.collide() or dist_tgt_end < 70:
                is_success = (dist_tgt_end < 70 and not env.collide())
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