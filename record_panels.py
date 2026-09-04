"""
하단 패널 왼쪽 3개 (LiDAR View, LiDAR Gauge View, LiDAR 1st View)를
동일한 주행 시점 동안 동시에 고화질 GIF로 추출하는 스크립트.
"""
import os
import sys
import time
import math
import subprocess
import random
import pygame
import numpy as np
import imageio_ffmpeg

from environment import BoatEnv
from perception import lidar_hits_np, update_grid, extract_clusters_from_grid, match_clusters
from navigation import find_gap, target_is_clear, is_direct_target_safe, is_waypoint_switch_safe, is_front_blocked
from utils import wrap, make_bezier_path, pure_pursuit

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

# 패널 정의: name -> (x, y, w, h, 파일명)
PANELS = {
    "panel_lidar_view": {
        "rect": (350, 665, 320, 220),
        "title": "LiDAR View",
        "mp4": "images/temp_lidar_view.mp4",
        "gif": "images/panel_lidar_view.gif"
    },
    "panel_gauge_view": {
        "rect": (700, 665, 320, 220),
        "title": "LiDAR Gauge View",
        "mp4": "images/temp_gauge_view.mp4",
        "gif": "images/panel_gauge_view.gif"
    },
    "panel_1st_view": {
        "rect": (1050, 665, 320, 220),
        "title": "LiDAR 1st View",
        "mp4": "images/temp_1st_view.mp4",
        "gif": "images/panel_1st_view.gif"
    }
}

def record_panels(speed_multiplier=4, required_laps=2, seed=42):
    print("========================================================")
    print(f"Recording Panels GIF: Speed={speed_multiplier}x, Laps={required_laps}, Seed={seed}")
    print("========================================================")
    
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs("images", exist_ok=True)
    
    # 2배 확대 해상도 (선명하고 깃허브 리드미에서 가독성 뛰어남)
    target_w, target_h = 640, 440
    video_fps = 16
    steps_per_frame = max(1, int(60 / video_fps * speed_multiplier / 4)) # 부드러운 프레임 캡처
    
    env = BoatEnv()
    env.sim_speed = speed_multiplier
    
    # 3개 패널용 FFmpeg 프로세스 시작
    procs = {}
    for key, pinfo in PANELS.items():
        cmd = [
            ffmpeg_exe, '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{target_w}x{target_h}',
            '-pix_fmt', 'rgb24',
            '-r', str(video_fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '16',
            '-pix_fmt', 'yuv420p',
            pinfo["mp4"]
        ]
        procs[key] = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    
    consecutive_successes = 0
    total_frames = 0
    sub_count = 0
    lap_steps = 0
    
    print("시뮬레이션 주행 및 3개 패널 동시 프레임 캡처 시작...")
    
    while consecutive_successes < required_laps:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                return
        
        env.frame += 1
        lap_steps += 1
        sub_count += 1
        
        env.update_dynamic_obstacles()
        dists, hits = lidar_hits_np(env.boat_pos, env.boat_heading, env.rel_angles, env.dynamic_obstacles, env.lidar_range)
        update_grid(env.grid, hits)
        env.grid *= 0.945
        
        new_c = extract_clusters_from_grid(env.grid)
        env.clusters, env.cluster_ids = match_clusters(env.clusters, env.cluster_ids, new_c)
        
        dist_to_target = np.linalg.norm(env.target - env.boat_pos)
        boat_spd = math.hypot(env.boat_vel[0], env.boat_vel[1])
        clear_to_target = is_direct_target_safe(env.boat_pos, env.boat_heading, env.target, env.dynamic_obstacles, env.boat_radius, boat_spd, params=env.params)
        
        if clear_to_target:
            new_wp = None
            env.current_wp = None
            env.next_wp = None
            env.candidate_wps = []
        else:
            new_wp = find_gap(env.clusters, env.cluster_ids, env.boat_pos, env.boat_heading, env.target, env.visited, env.grid, env.dynamic_obstacles, params=env.params)
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
                env.candidate_wps = []
                
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
                if new_wp is not None and (new_wp["pair"] == env.current_wp["pair"] or new_wp["pair"] == (id2, id1)):
                    env.current_wp["score"] = new_wp["score"]
                    if "factors" in new_wp:
                        env.current_wp["factors"] = new_wp["factors"]
            elif new_wp is not None:
                if is_waypoint_switch_safe(env.boat_pos, env.boat_heading, env.current_wp["pos"], new_wp["pos"], env.dynamic_obstacles, env.boat_radius, boat_spd, params=env.params):
                    env.current_wp = new_wp
                    
        if new_wp is not None:
            if env.current_wp is None:
                env.current_wp = new_wp
            elif new_wp["pair"] != env.current_wp["pair"] and new_wp["pair"] != (env.current_wp["pair"][1], env.current_wp["pair"][0]):
                dist_to_curr = np.linalg.norm(env.current_wp["pos"] - env.boat_pos)
                front_blocked = is_front_blocked(env.boat_pos, env.boat_heading, env.dynamic_obstacles, env.boat_radius, block_dist=120.0, fov_deg=65.0)
                if not front_blocked and dist_to_curr > 80:
                    threshold = float(env.params.get('wp_switch_thresh', 1.1))
                    if new_wp["score"] > env.current_wp["score"] * threshold:
                        if is_waypoint_switch_safe(env.boat_pos, env.boat_heading, env.current_wp["pos"], new_wp["pos"], env.dynamic_obstacles, env.boat_radius, boat_spd, params=env.params):
                            env.current_wp = new_wp
                            
        if env.current_wp is not None and not clear_to_target:
            temp_visited = env.visited.copy()
            temp_visited.add(env.current_wp["pair"])
            temp_visited.add((env.current_wp["pair"][1], env.current_wp["pair"][0]))
            vec = env.current_wp["pos"] - env.boat_pos
            next_head = math.atan2(vec[1], vec[0])
            env.next_wp = find_gap(env.clusters, env.cluster_ids, env.current_wp["pos"], next_head, env.target, temp_visited, env.grid, env.dynamic_obstacles, params=env.params)
        else:
            env.next_wp = None
            env.next_bezier_path = None
            
        goal = env.current_wp["pos"] if (env.current_wp is not None and not clear_to_target) else env.target
        env.bezier_path = make_bezier_path(env.boat_pos, env.boat_heading, goal, obstacles=env.dynamic_obstacles, boat_radius=env.boat_radius, boat_speed=boat_spd)
        if env.bezier_path is not None:
            env.pursuit_target = pure_pursuit(env.bezier_path, env.boat_pos, lookahead=70)
            
        steer = env.update_steering(dists)
        if steer is None:
            steer = 0
            
        L, R = env.get_pwm(steer)
        env.step(L, R)
        
        # 프레임 캡처 주기
        if sub_count >= steps_per_frame:
            sub_count = 0
            env.render(hits)
            
            # 각 패널 영역 서브서피스 추출 & 리사이즈 & 파이프 전송
            for key, pinfo in PANELS.items():
                cx, cy, cw, ch = pinfo["rect"]
                sub = env.screen.subsurface(pygame.Rect(cx, cy, cw, ch))
                scaled = pygame.transform.smoothscale(sub, (target_w, target_h))
                frame_bytes = pygame.image.tostring(scaled, 'RGB')
                procs[key].stdin.write(frame_bytes)
                
            total_frames += 1
            
        is_reached = (np.linalg.norm(env.target - env.boat_pos) < 70)
        is_collide = env.collide()
        
        if is_collide or is_reached:
            if is_reached and not is_collide:
                consecutive_successes += 1
                print(f"  완주 성공! [{consecutive_successes}/{required_laps}] (steps: {lap_steps})", flush=True)
            else:
                print(f"  충돌 발생! 카운트 리셋 (0/{required_laps})", flush=True)
                consecutive_successes = 0
            lap_steps = 0
            env.reset()
            env.sim_speed = speed_multiplier
            
    # 파이프 종료
    for key, proc in procs.items():
        proc.stdin.close()
        proc.wait()
        
    print(f"녹화 완료 (총 {total_frames} 프레임). GIF 변환 시작...")
    pygame.quit()
    
    # 각 패널별 고품질 GIF 변환
    for key, pinfo in PANELS.items():
        mp4_file = pinfo["mp4"]
        gif_file = pinfo["gif"]
        print(f"  변환 중: {pinfo['title']} -> {gif_file}")
        
        cmd_gif = [
            ffmpeg_exe, '-y',
            '-i', mp4_file,
            '-vf', f'fps={video_fps},scale={target_w}:{target_h}:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3',
            gif_file
        ]
        subprocess.run(cmd_gif, check=True)
        
        if os.path.exists(mp4_file):
            os.remove(mp4_file)
            
        sz = os.path.getsize(gif_file) / (1024 * 1024)
        print(f"    완료: {gif_file} ({sz:.2f}MB)", flush=True)
        
    print("\n모든 패널 GIF 생성 완료!")

if __name__ == "__main__":
    record_panels(speed_multiplier=4, required_laps=2, seed=42)
