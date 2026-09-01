"""
train_perfect_nav.py - 초극초고속 배속 100회 무충돌 자동 강화학습 최적화 루프
(웨이포인트 도달거리와 이멀전시 감지거리는 고정하고 조타/모멘트/회피력 집중 튜닝)
"""
import pygame
import numpy as np
import math
import sys
import copy
import time
import json
from environment import BoatEnv
from perception import lidar_hits_np, update_grid, extract_clusters_from_grid, match_clusters
from navigation import find_gap, target_is_clear
from utils import wrap, make_bezier_path, pure_pursuit

def evaluate_candidate(env, params, n_episodes=100, max_frames=2600, render_every=80):
    """후보 파라미터 세트로 N개 무작위 맵 에피소드를 초극초고속 배속 실행하고 결과 반환"""
    goals = 0
    collisions = 0
    timeouts = 0
    frame_counts = []
    
    # 핵심 튜닝 파라미터 언팩
    steer_gain = params['steer_gain']
    steer_alpha = params['steer_alpha']
    mom_coeff = params['mom_coeff']
    pwm_rng = params['pwm_rng']
    avoid_normal = params['avoid_normal']
    avoid_em = params['avoid_em']
    clear_margin = params['clear_margin']
    
    # 고정 파라미터
    em_dist = 70.0
    wp_arrive = 25.0

    for ep in range(n_episodes):
        env.reset()
        ep_result = 'timeout'
        
        for frame in range(max_frames):
            env.frame += 1
            env.update_dynamic_obstacles()

            if frame % 50 == 0:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        return goals, collisions, timeouts, 9999, True

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
            env.validate_wp_grid()
            env.validate_wp_obstacle_5x5()

            if target_is_clear(env.boat_pos, env.target, env.dynamic_obstacles, boat_radius=25 + clear_margin):
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

            if env.current_wp is not None:
                should_clear = False
                vec_to_wp = env.current_wp["pos"] - env.boat_pos
                dnow = np.linalg.norm(vec_to_wp)
                if dnow < wp_arrive:
                    should_clear = True
                wp_angle = math.atan2(vec_to_wp[1], vec_to_wp[0])
                angle_diff = abs(wrap(wp_angle - env.boat_heading))
                if angle_diff > np.pi / 2 and dnow < 60:
                    should_clear = True
                if should_clear:
                    p = env.current_wp["pair"]
                    env.visited.add(p)
                    env.visited.add((p[1], p[0]))
                    env.current_wp = None

            if new_wp is not None:
                if env.current_wp is None:
                    env.current_wp = new_wp
                else:
                    dist_to_curr = np.linalg.norm(env.current_wp["pos"] - env.boat_pos)
                    if dist_to_curr > 80:
                        if new_wp["score"] > env.current_wp["score"] * 1.15:
                            env.current_wp = new_wp

            if env.current_wp is not None:
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
                    goal = env.target
                else:
                    goal = env.current_wp["pos"]
                env.bezier_path = make_bezier_path(env.boat_pos, env.boat_heading, goal)
                if env.bezier_path is not None:
                    env.pursuit_target = pure_pursuit(env.bezier_path, env.boat_pos, lookahead=70)
                if env.current_wp is not None and env.next_wp is not None:
                    vec = env.current_wp["pos"] - env.boat_pos
                    next_start_head = math.atan2(vec[1], vec[0])
                    env.next_bezier_path = make_bezier_path(env.current_wp["pos"], next_start_head, env.next_wp["pos"])
                    if env.next_bezier_path is not None:
                        env.next_pursuit_target = pure_pursuit(env.next_bezier_path, env.current_wp["pos"], lookahead=70)
                else:
                    env.next_bezier_path = None
                    env.next_pursuit_target = None

            visual_target = env.pursuit_target
            if env.current_wp is not None and env.next_pursuit_target is not None and env.pursuit_target is not None:
                dist_to_wp = np.linalg.norm(env.current_wp["pos"] - env.boat_pos)
                if dist_to_wp < 75:
                    env.pursuit_target = env.next_pursuit_target

            # 조타 연산
            center_idx = env.lidar_beams // 2
            span = env.lidar_beams // 12
            front_dists = dists[center_idx - span : center_idx + span]
            min_front_dist = np.min(front_dists)
            
            is_emergency = min_front_dist < em_dist
            
            if env.pursuit_target is not None:
                px, py = env.pursuit_target
                heading_target = math.atan2(py - env.boat_pos[1], px - env.boat_pos[0])
                heading_error = wrap(heading_target - env.boat_heading)
                steer_raw = heading_error * steer_gain
                steer_f = steer_alpha * steer_raw + (1 - steer_alpha) * env.prev_steer
                env.prev_steer = steer_f
                
                from navigation import reactive_avoidance
                avoid = reactive_avoidance(dists, env.rel_angles)
                avoid_multiplier = avoid_em if is_emergency else avoid_normal
                steer = float(np.clip(steer_f + avoid_multiplier * avoid, -1, 1))
            else:
                steer = 0

            # PWM 계산
            dead = 0.02
            st = steer if abs(steer) >= dead else 0
            mid = 1500
            m = np.log1p(3 * abs(st)) / np.log(4)
            d = m * pwm_rng
            if st >= 0: L = mid - d; R = mid + d
            else: L = mid + d; R = mid - d
            L = int(np.clip(L, 1100, 1900))
            R = int(np.clip(R, 1100, 1900))

            # Step 물리
            tL = L * 10
            tR = R * 10
            target_fwd = (tL + tR) / 9.0
            if is_emergency:
                target_fwd = 0.0
            if not hasattr(env, 'current_fwd'):
                env.current_fwd = 0.0
            env.current_fwd = env.current_fwd * 0.95 + target_fwd * 0.05
            
            mom = (tR - tL) * mom_coeff
            hv = np.array([math.cos(env.boat_heading), math.sin(env.boat_heading)])
            acc = env.current_fwd / env.mass
            vel_norm = np.linalg.norm(env.boat_vel)
            drag = -env.drag * vel_norm * env.boat_vel if vel_norm > 0 else np.zeros(2)
            
            env.boat_vel += (acc * hv + drag) * env.dt
            env.boat_pos += env.boat_vel * env.dt
            
            ang_acc = (mom - env.rot_drag * env.boat_ang_vel) / env.inertia
            env.boat_ang_vel += ang_acc * env.dt
            env.boat_ang_vel *= 0.84
            env.boat_heading += env.boat_ang_vel * env.dt

            # 초극초고속 배속 렌더링 (80프레임마다 한 번만 렌더링)
            if frame % render_every == 0:
                env.render(hits)

            dist_to_target = np.linalg.norm(env.target - env.boat_pos)
            if env.collide():
                ep_result = 'collide'
                collisions += 1
                break
            if dist_to_target < 70:
                ep_result = 'goal'
                goals += 1
                frame_counts.append(frame)
                break

        if ep_result == 'timeout':
            timeouts += 1

    avg_f = np.mean(frame_counts) if frame_counts else 9999
    return goals, collisions, timeouts, avg_f, False

def mutate_params(base, scale=0.08):
    """현재 최고 파라미터 주변에서 집중 변이(Mutation) 탐색"""
    p = copy.deepcopy(base)
    
    p['steer_gain'] = float(np.clip(p['steer_gain'] + np.random.normal(0, 0.03 * scale * 10), 0.65, 0.95))
    p['steer_alpha'] = float(np.clip(p['steer_alpha'] + np.random.normal(0, 0.02 * scale * 10), 0.30, 0.45))
    p['mom_coeff'] = float(np.clip(p['mom_coeff'] + np.random.normal(0, 0.0003 * scale * 10), 0.0055, 0.0080))
    p['pwm_rng'] = float(np.clip(p['pwm_rng'] + np.random.normal(0, 8 * scale * 10), 240, 310))
    p['avoid_normal'] = float(np.clip(p['avoid_normal'] + np.random.normal(0, 0.002 * scale * 10), 0.015, 0.035))
    p['avoid_em'] = float(np.clip(p['avoid_em'] + np.random.normal(0, 0.012 * scale * 10), 0.08, 0.18))
    p['clear_margin'] = float(np.clip(p['clear_margin'] + np.random.normal(0, 0.8 * scale * 10), 1.0, 7.0))
    
    return p

def main():
    env = BoatEnv()
    
    # 90% 성적을 낸 최적 베이스라인
    best_params = {
        'steer_gain': 0.75,
        'steer_alpha': 0.35,
        'mom_coeff': 0.0065,
        'pwm_rng': 270.0,
        'avoid_normal': 0.020,
        'avoid_em': 0.10,
        'clear_margin': 2.0
    }
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   KABOAT 초극초고속 배속 100회 무충돌 달성 자동 최적화 루프  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(" [조건] wp도달거리(25px) 및 em감지거리(70px) 고정")
    print(" [목표] 100회 무작위 맵 연속 100.0% 무충돌 완주 달성 시까지 자동 학습")
    print("──────────────────────────────────────────────────────────────")

    best_score = -1.0
    best_goals = 0
    generation = 0

    while True:
        generation += 1
        
        if generation == 1:
            candidate = copy.deepcopy(best_params)
        else:
            candidate = mutate_params(best_params, scale=0.08)

        t0 = time.time()
        goals, collisions, timeouts, avg_f, quit_flag = evaluate_candidate(
            env, candidate, n_episodes=100, max_frames=2600, render_every=80
        )
        t_elapsed = time.time() - t0

        if quit_flag:
            print("\n[사용자 중단] 시뮬레이션을 종료합니다.")
            break

        score = (goals * 100.0) - (collisions * 150.0) - (timeouts * 50.0) - (avg_f * 0.01)

        is_new_best = goals > best_goals or (goals == best_goals and score > best_score)
        
        status_tag = "🔥 NEW BEST!" if is_new_best else "  "
        print(f"Gen {generation:03d} | 도달: {goals:3d}/100 ({goals}%) | 충돌: {collisions:2d} | 타임아웃: {timeouts:2d} | 평균: {avg_f:.0f}f | {t_elapsed:.1f}s {status_tag}")

        if is_new_best:
            best_score = score
            best_goals = goals
            best_params = copy.deepcopy(candidate)
            print(f"   ↳ [최고 갱신] 도달률: {goals}% | params: {best_params}")

        # 100% 무충돌 달성 시 즉시 종료 및 파일 저장
        if goals == 100:
            print("\n" + "★" * 60)
            print("  🏆 100회 무작위 맵 연속 100.0% 무충돌 완주 달성 성공! 🏆")
            print("★" * 60)
            print(f"  최종 최적 파라미터 세트:")
            for k, v in best_params.items():
                print(f"    • {k:18s}: {v:.4f}")
            print("──────────────────────────────────────────────────────────────")
            
            with open("best_learned_params.json", "w") as f:
                json.dump(best_params, f, indent=2)
            print(f"[완료] best_learned_params.json 저장 완료.")
            break

    pygame.quit()

if __name__ == "__main__":
    main()
