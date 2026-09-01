import os
import sys
import time
import math
import random
import json
import datetime
import pygame
import numpy as np

from environment import BoatEnv
from perception import lidar_hits_np, update_grid, extract_clusters_from_grid, match_clusters
from navigation import find_gap, target_is_clear
from utils import wrap, make_bezier_path, pure_pursuit

def run_overnight():
    pygame.init()
    
    env = BoatEnv()
    # 2배속 설정
    env.sim_speed = 2
    
    total_episodes = 1000
    completed_episodes = 0
    success_count = 0
    collision_count = 0
    timeout_count = 0
    
    start_time = time.time()
    results = []
    
    log_file = "overnight_1000_details.log"
    txt_report_file = "overnight_1000_result.txt"
    json_report_file = "overnight_1000_result.json"
    
    f_log = open(log_file, "w")
    f_log.write(f"=== 1000-RUN OVERNIGHT EVALUATION LOG ===\n")
    f_log.write(f"Started At: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f_log.write("=" * 60 + "\n\n")
    f_log.flush()
    
    # 폰트
    font = pygame.font.SysFont(None, 24)
    bold_font = pygame.font.SysFont(None, 26, bold=True)
    small_font = pygame.font.SysFont(None, 18)
    
    score_surf = pygame.Surface((440, 95), pygame.SRCALPHA)
    
    running = True
    ep_frame = 0
    max_ep_frames = 4000
    
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
                break
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:
                    env.handle_click(e.pos)

        if completed_episodes >= total_episodes:
            # 1,000회 모두 완료된 상태: 최종 결과 고정 화면 렌더링
            env.screen.fill((20, 35, 60))
            box_w, box_h = 600, 300
            bx = (env.w - box_w) // 2
            by = (env.h - box_h) // 2
            
            pygame.draw.rect(env.screen, (10, 20, 40), (bx, by, box_w, box_h))
            pygame.draw.rect(env.screen, (50, 220, 120), (bx, by, box_w, box_h), 3)
            
            sr = (success_count / total_episodes) * 100.0
            cr = (collision_count / total_episodes) * 100.0
            tr = (timeout_count / total_episodes) * 100.0
            elapsed = time.time() - start_time
            
            t1 = bold_font.render("=== 1,000-RUN OVERNIGHT EVALUATION COMPLETED ===", True, (255, 230, 80))
            t2 = bold_font.render(f"Total Episodes: {total_episodes} (2X SPEED)", True, (220, 235, 255))
            t3 = bold_font.render(f"SUCCESS: {success_count} / {total_episodes} ({sr:.2f}%)", True, (50, 240, 100))
            t4 = bold_font.render(f"COLLISION: {collision_count} / {total_episodes} ({cr:.2f}%)", True, (255, 80, 80))
            t5 = bold_font.render(f"TIMEOUT: {timeout_count} / {total_episodes} ({tr:.2f}%)", True, (240, 180, 50))
            t6 = font.render(f"Elapsed Time: {elapsed/60.0:.1f} min | Saved to overnight_1000_result.txt", True, (180, 210, 240))
            
            env.screen.blit(t1, (bx + 30, by + 30))
            env.screen.blit(t2, (bx + 30, by + 75))
            env.screen.blit(t3, (bx + 30, by + 115))
            env.screen.blit(t4, (bx + 30, by + 155))
            env.screen.blit(t5, (bx + 30, by + 195))
            env.screen.blit(t6, (bx + 30, by + 245))
            
            pygame.display.flip()
            env.clock.tick(30)
            continue

        # 2배속 서브스텝 실행
        sub_steps = max(1, int(getattr(env, 'sim_speed', 2)))
        hits = None
        ep_ended = False
        outcome = None
        
        for _ in range(sub_steps):
            ep_frame += 1
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

            # 판정
            if env.collide():
                ep_ended = True
                outcome = 'COLLISION'
                collision_count += 1
                break
            elif np.linalg.norm(env.target - env.boat_pos) < 70:
                ep_ended = True
                outcome = 'SUCCESS'
                success_count += 1
                break
            elif ep_frame >= max_ep_frames:
                ep_ended = True
                outcome = 'TIMEOUT'
                timeout_count += 1
                break

        if ep_ended:
            completed_episodes += 1
            final_dist = float(np.linalg.norm(env.target - env.boat_pos))
            cur_sr = (success_count / completed_episodes) * 100.0
            cur_cr = (collision_count / completed_episodes) * 100.0
            
            res_item = {
                'episode': completed_episodes,
                'outcome': outcome,
                'frames': ep_frame,
                'final_dist': round(final_dist, 1)
            }
            results.append(res_item)
            
            log_line = f"[{completed_episodes:4d}/{total_episodes}] {outcome:9s} | Frames: {ep_frame:4d} | Dist: {final_dist:5.1f}px | SR: {cur_sr:5.1f}% | CR: {cur_cr:5.1f}%"
            f_log.write(log_line + "\n")
            f_log.flush()
            print(log_line)
            
            # 중간 보고서 실시간 갱신 (10회마다 or 최종)
            if completed_episodes % 5 == 0 or completed_episodes == total_episodes:
                elapsed_sec = time.time() - start_time
                summary_data = {
                    'total_episodes': total_episodes,
                    'completed_episodes': completed_episodes,
                    'success_count': success_count,
                    'collision_count': collision_count,
                    'timeout_count': timeout_count,
                    'success_rate_percent': round(cur_sr, 2),
                    'collision_rate_percent': round(cur_cr, 2),
                    'timeout_rate_percent': round((timeout_count / completed_episodes) * 100.0, 2),
                    'avg_frames': round(float(np.mean([r['frames'] for r in results])), 1),
                    'elapsed_minutes': round(elapsed_sec / 60.0, 1),
                    'status': 'COMPLETED' if completed_episodes >= total_episodes else 'IN_PROGRESS',
                    'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                with open(json_report_file, "w") as f_json:
                    json.dump(summary_data, f_json, indent=2)
                    
                with open(txt_report_file, "w") as f_txt:
                    f_txt.write("=" * 60 + "\n")
                    f_txt.write("       1,000-RUN OVERNIGHT EVALUATION REPORT\n")
                    f_txt.write("=" * 60 + "\n")
                    f_txt.write(f"Status: {summary_data['status']}\n")
                    f_txt.write(f"Progress: {completed_episodes} / {total_episodes} ({completed_episodes/total_episodes*100:.1f}%)\n")
                    f_txt.write(f"Success Rate:   {summary_data['success_rate_percent']} %  ({success_count} wins)\n")
                    f_txt.write(f"Collision Rate: {summary_data['collision_rate_percent']} %  ({collision_count} crashes)\n")
                    f_txt.write(f"Timeout Rate:   {summary_data['timeout_rate_percent']} %  ({timeout_count} timeouts)\n")
                    f_txt.write(f"Average Frames: {summary_data['avg_frames']} frames/episode\n")
                    f_txt.write(f"Elapsed Time:   {summary_data['elapsed_minutes']} minutes\n")
                    f_txt.write(f"Last Updated:   {summary_data['updated_at']}\n")
                    f_txt.write("=" * 60 + "\n")
            
            ep_frame = 0
            env.reset()

        # 화면 렌더링
        if hits is not None:
            env.render(hits)
            
        # 상단 실시간 오버나이트 스코어보드 오버레이
        sw, sh = 440, 95
        sx = (env.w - sw) // 2
        sy = 12
        score_surf.fill((10, 22, 45, 225))
        pygame.draw.rect(score_surf, (0, 210, 255), (0, 0, sw, sh), 2)
        
        curr_ep = min(total_episodes, completed_episodes + 1)
        sr = (success_count / completed_episodes * 100.0) if completed_episodes > 0 else 0.0
        cr = (collision_count / completed_episodes * 100.0) if completed_episodes > 0 else 0.0
        
        title_t = bold_font.render(f"OVERNIGHT 1000 EVALUATION ({env.sim_speed}X SPEED)", True, (255, 230, 80))
        ep_t = small_font.render(f"Episode: {curr_ep}/{total_episodes}", True, (200, 235, 255))
        score_surf.blit(title_t, (12, 8))
        score_surf.blit(ep_t, (sw - ep_t.get_width() - 12, 10))
        
        s_txt = bold_font.render(f"SUCCESS: {success_count} ({sr:.1f}%)", True, (40, 240, 110))
        c_txt = bold_font.render(f"COLLISION: {collision_count} ({cr:.1f}%)", True, (255, 85, 75))
        score_surf.blit(s_txt, (12, 34))
        score_surf.blit(c_txt, (220, 34))
        
        # 하단 프로그레스 바
        bx_p, by_p, bw_p, bh_p = 12, 65, sw - 24, 16
        pygame.draw.rect(score_surf, (25, 40, 60), (bx_p, by_p, bw_p, bh_p))
        pygame.draw.rect(score_surf, (80, 120, 170), (bx_p, by_p, bw_p, bh_p), 1)
        
        if completed_episodes > 0:
            sw_bar = int((success_count / total_episodes) * bw_p)
            cw_bar = int((collision_count / total_episodes) * bw_p)
            if sw_bar > 0:
                pygame.draw.rect(score_surf, (40, 220, 100), (bx_p, by_p, sw_bar, bh_p))
            if cw_bar > 0:
                pygame.draw.rect(score_surf, (240, 70, 60), (bx_p + sw_bar, by_p, cw_bar, bh_p))
                
        env.screen.blit(score_surf, (sx, sy))
        pygame.display.flip()
        
        env.clock.tick(60)

    f_log.close()
    pygame.quit()

if __name__ == '__main__':
    run_overnight()
