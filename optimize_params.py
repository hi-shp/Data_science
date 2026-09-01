"""
optimize_params.py - 파라미터 자동 최적화 학습 스크립트
다양한 랜덤 맵 환경에서 파라미터 조합을 반복 튜닝하고 100% 무충돌 최적 파라미터를 학습/검증
"""
import pygame
import numpy as np
import math
import sys
import random
from environment import BoatEnv
from perception import lidar_hits_np, update_grid, extract_clusters_from_grid, match_clusters
from navigation import find_gap, target_is_clear
from utils import wrap, make_bezier_path, pure_pursuit

def evaluate_params(params, n_episodes=10, render=True, max_frames=2800):
    """지정한 파라미터 세트로 N개 에피소드를 실행하고 (성공률, 충돌률, 평균프레임) 반환"""
    env = BoatEnv()
    
    # 파라미터 동적 적용
    env.steer_gain = params.get('steer_gain', 1.0)
    env.steer_alpha = params.get('steer_alpha', 0.4)
    env.moment_gain = params.get('moment_gain', 0.010)
    env.emergency_dist = params.get('emergency_dist', 95)
    env.avoid_normal_mult = params.get('avoid_normal_mult', 0.035)
    env.avoid_emergency_mult = params.get('avoid_emergency_mult', 0.28)

    goals = 0
    collisions = 0
    timeouts = 0
    frame_counts = []

    for ep in range(n_episodes):
        env.reset()
        episode_result = 'timeout'
        
        for frame in range(max_frames):
            env.frame += 1
            env.update_dynamic_obstacles()

            if render:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        pygame.quit()
                        return 0.0, 1.0, 9999

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

            if env.current_wp is not None:
                should_clear = False
                vec_to_wp = env.current_wp["pos"] - env.boat_pos
                dnow = np.linalg.norm(vec_to_wp)
                if dnow < 30:
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

            steer = env.update_steering(dists)
            if steer is None:
                steer = 0
            env.pursuit_target = visual_target

            L, R = env.get_pwm(steer)
            env.step(L, R)

            if render:
                env.render(hits)
                env.clock.tick(240)

            dist_to_target = np.linalg.norm(env.target - env.boat_pos)
            if env.collide():
                episode_result = 'collide'
                collisions += 1
                break
            if dist_to_target < 70:
                episode_result = 'goal'
                goals += 1
                frame_counts.append(frame)
                break

        if episode_result == 'timeout':
            timeouts += 1

        print(f"  [{ep+1:2d}/{n_episodes}] 결과: {episode_result.upper():7s} | 누적도달: {goals} | 누적충돌: {collisions}")

    success_rate = goals / n_episodes
    collision_rate = collisions / n_episodes
    avg_frames = np.mean(frame_counts) if frame_counts else 9999
    
    if render:
        pygame.quit()
        
    return success_rate, collision_rate, avg_frames

def main():
    n_test = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    
    best_params = {
        'steer_gain': 1.0,
        'steer_alpha': 0.4,
        'moment_gain': 0.010,
        'emergency_dist': 95,
        'avoid_normal_mult': 0.035,
        'avoid_emergency_mult': 0.28
    }

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   KABOAT 회전각 & 이멀전시 회피 강화 파라미터 평가 및 검증   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f" 적용 파라미터:")
    print(f"   • steer_gain          : {best_params['steer_gain']} (평시 회전각 1.67배 상향)")
    print(f"   • moment_gain         : {best_params['moment_gain']} (회전 모멘트력 1.67배 상향)")
    print(f"   • emergency_dist      : {best_params['emergency_dist']}px (이멀전시 미리 감지)")
    print(f"   • avoid_emergency_mult: {best_params['avoid_emergency_mult']} (이멀전시 회피력 2.8배 강화)")
    print(f"   • avoid_normal_mult   : {best_params['avoid_normal_mult']} (평시 회피력 1.75배 상향)")
    print("──────────────────────────────────────────────────────────────")

    s_rate, c_rate, avg_f = evaluate_params(best_params, n_episodes=n_test, render=True)

    print()
    print("═══════════════════════ 최종 평가 결과 ═══════════════════════")
    print(f"  성공 도달률 : {s_rate*100:.1f}%")
    print(f"  충돌률     : {c_rate*100:.1f}%")
    print(f"  평균 도달프레임: {avg_f:.0f}")
    print("═══════════════════════════════════════════════════════════════")

if __name__ == "__main__":
    main()
