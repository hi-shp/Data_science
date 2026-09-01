"""
tune_params.py - 시각적 자동 파라미터 검증 스크립트
N회 랜덤 맵에서 화면에 띄워서 충돌률/도달률 측정
"""
import pygame
import numpy as np
import math
import sys
from environment import BoatEnv
from perception import lidar_hits_np, update_grid, extract_clusters_from_grid, match_clusters
from navigation import find_gap, target_is_clear
from utils import wrap, make_bezier_path, pure_pursuit

def run_episode(env, max_frames=3000):
    """한 에피소드 실행 (화면 렌더링 포함), 결과 반환"""
    env.reset()
    
    for frame in range(max_frames):
        env.frame += 1
        env.update_dynamic_obstacles()
        
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                return 'quit', frame
        
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

        if new_wp is not None:
            if env.current_wp is None:
                env.current_wp = new_wp
            else:
                dist_to_curr = np.linalg.norm(env.current_wp["pos"] - env.boat_pos)
                if dist_to_curr > 80:
                    vec_curr = env.current_wp["pos"] - env.boat_pos
                    vec_new = new_wp["pos"] - env.boat_pos
                    ang_curr = math.atan2(vec_curr[1], vec_curr[0])
                    ang_new = math.atan2(vec_new[1], vec_new[0])
                    angle_diff = abs(wrap(ang_new - ang_curr))
                    threshold = 1.1
                    if new_wp["score"] > env.current_wp["score"] * threshold:
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
            if dist_to_wp < 85:
                env.pursuit_target = env.next_pursuit_target

        steer = env.update_steering(dists)
        if steer is None:
            steer = 0
        env.pursuit_target = visual_target

        L, R = env.get_pwm(steer)
        env.step(L, R)
        
        # 화면 렌더링
        env.render(hits)
        env.clock.tick(240)
        
        dist_to_target = np.linalg.norm(env.target - env.boat_pos)
        if env.collide():
            return 'collide', frame
        if dist_to_target < 70:
            return 'goal', frame
    
    return 'timeout', max_frames

def main():
    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    
    env = BoatEnv()
    
    print(f"╔══════════════════════════════════════════════╗")
    print(f"║   KABOAT 시각적 검증 ({n_episodes}회 테스트)           ║")
    print(f"╚══════════════════════════════════════════════╝")
    print()
    
    goals = 0
    collisions = 0
    timeouts = 0
    total_frames = []
    
    for i in range(n_episodes):
        result, frames = run_episode(env, max_frames=3000)
        
        if result == 'quit':
            print("\n  [종료] 사용자가 창을 닫았습니다.")
            break
        elif result == 'goal':
            goals += 1
            total_frames.append(frames)
            status = f"✅ 도달 ({frames}f)"
        elif result == 'collide':
            collisions += 1
            status = f"💥 충돌 ({frames}f)"
        else:
            timeouts += 1
            status = f"⏰ 타임아웃"
        
        done = i + 1
        pct = done / n_episodes * 100
        print(f"  [{done:3d}/{n_episodes}] {status}  |  도달:{goals}  충돌:{collisions}  타임아웃:{timeouts}  ({pct:.0f}%)")
    
    print()
    done = goals + collisions + timeouts
    if done > 0:
        print(f"══════════════════ 최종 결과 ══════════════════")
        print(f"  목표 도달률: {goals}/{done} ({goals/done*100:.1f}%)")
        print(f"  충돌률:     {collisions}/{done} ({collisions/done*100:.1f}%)")
        print(f"  타임아웃:   {timeouts}/{done} ({timeouts/done*100:.1f}%)")
        if total_frames:
            print(f"  평균 도달 프레임: {np.mean(total_frames):.0f} (최소:{min(total_frames)} 최대:{max(total_frames)})")
        print(f"═══════════════════════════════════════════════")
    
    pygame.quit()

if __name__ == "__main__":
    main()
