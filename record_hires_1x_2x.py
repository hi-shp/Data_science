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

def draw_mouse_cursor(surface, pos, clicking=False, click_progress=0.0):
    x, y = int(pos[0]), int(pos[1])
    if clicking:
        radius = int(8 + click_progress * 22)
        alpha = int(255 * (1.0 - click_progress))
        ripple_surf = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(ripple_surf, (255, 230, 50, alpha), (radius + 2, radius + 2), radius, 2)
        surface.blit(ripple_surf, (x - radius - 2, y - radius - 2))
        pygame.draw.circle(surface, (255, 240, 80), (x, y), 5)
    
    cursor_pts = [
        (x, y),
        (x, y + 16),
        (x + 5, y + 13),
        (x + 9, y + 19),
        (x + 12, y + 17),
        (x + 8, y + 11),
        (x + 13, y + 11)
    ]
    pygame.draw.polygon(surface, (15, 15, 15), [(px, py) for px, py in cursor_pts])
    pygame.draw.polygon(surface, (255, 255, 255), [(px + 1, py + 1) for px, py in cursor_pts[:-1]])

def record_hires_run(speed_multiplier, seed=42, required_laps=3, out_gif=None):
    print(f"\n========================================================")
    print(f"Recording High-Res {speed_multiplier}x Run (Seed={seed}, {required_laps} Laps)")
    print(f"========================================================")
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Video physics timing:
    # Standard: 60 physics steps = 1 real second.
    # At 1x: 60 steps/sec. With video at 30 fps -> 2 steps per frame.
    # At 2x: 120 steps/sec. With video at 30 fps -> 4 steps per frame.
    video_fps = 30
    steps_per_frame = 2 * speed_multiplier
    
    env = BoatEnv()
    env.sim_speed = 1
    
    target_w, target_h = 1280, 640
    raw_mp4 = f"images/temp_hires_{speed_multiplier}x.mp4"
    
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
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        raw_mp4
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    
    btn_rect = env.speed_btns[speed_multiplier]
    btn_center = np.array([btn_rect.centerx, btn_rect.centery], dtype=float)
    start_cursor = np.array([btn_rect.centerx - 65, btn_rect.centery + 50], dtype=float)
    
    intro_frames = 20
    click_frame = 10
    for f in range(intro_frames):
        dists, hits = lidar_hits_np(env.boat_pos, env.boat_heading, env.rel_angles, env.dynamic_obstacles, env.lidar_range)
        env.render(hits)
        
        if f < click_frame:
            t = f / float(click_frame)
            cur_pos = start_cursor * (1.0 - t) + btn_center * t
            draw_mouse_cursor(env.screen, cur_pos, clicking=False)
        elif f == click_frame:
            env.sim_speed = speed_multiplier
            env.render(hits)
            draw_mouse_cursor(env.screen, btn_center, clicking=True, click_progress=0.1)
        else:
            prog = (f - click_frame) / float(intro_frames - click_frame)
            draw_mouse_cursor(env.screen, btn_center, clicking=True, click_progress=prog)
            
        scaled_surf = pygame.transform.smoothscale(env.screen, (target_w, target_h))
        frame_bytes = pygame.image.tostring(scaled_surf, 'RGB')
        proc.stdin.write(frame_bytes)
        
    print(f"Button {speed_multiplier}x clicked! Running {required_laps} consecutive laps...")
    
    consecutive_successes = 0
    total_frames = intro_frames
    lap_steps = 0
    sub_count = 0
    
    while consecutive_successes < required_laps:
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
            env.current_wp = None
            env.next_wp = None
            env.candidate_wps = []
        else:
            new_wp = find_gap(env.clusters, env.cluster_ids, env.boat_pos, env.boat_heading, env.target, env.visited, env.grid, env.dynamic_obstacles, params=env.params)
            if new_wp is not None:
                env.candidate_wps = new_wp.get("candidates", [])
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
            elif env.current_wp is not None:
                env.candidate_wps = env.current_wp.get("candidates", [])
            else:
                env.candidate_wps = []
                
        if env.current_wp is not None:
            if np.linalg.norm(env.current_wp["pos"] - env.boat_pos) < 25 or target_is_clear(env.boat_pos, env.target, env.dynamic_obstacles):
                p = env.current_wp["pair"]
                env.visited.add(p)
                env.visited.add((p[1], p[0]))
                env.current_wp = None
                env.candidate_wps = []
                
        if env.current_wp is not None and not clear_to_target:
            temp_visited = env.visited.copy()
            temp_visited.add(env.current_wp["pair"])
            temp_visited.add((env.current_wp["pair"][1], env.current_wp["pair"][0]))
            vec = env.current_wp["pos"] - env.boat_pos
            next_head = math.atan2(vec[1], vec[0])
            env.next_wp = find_gap(env.clusters, env.cluster_ids, env.current_wp["pos"], next_head, env.target, temp_visited, env.grid, env.dynamic_obstacles, params=env.params)
            if env.next_wp is not None:
                env.next_bezier_path = make_bezier_path(env.current_wp["pos"], next_head, env.next_wp["pos"], obstacles=env.dynamic_obstacles, boat_radius=env.boat_radius, boat_speed=boat_spd)
            else:
                env.next_bezier_path = None
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
        
        if sub_count >= steps_per_frame:
            sub_count = 0
            env.render(hits)
            scaled_surf = pygame.transform.smoothscale(env.screen, (target_w, target_h))
            frame_bytes = pygame.image.tostring(scaled_surf, 'RGB')
            proc.stdin.write(frame_bytes)
            total_frames += 1
            
        is_reached = (np.linalg.norm(env.target - env.boat_pos) < 70)
        is_collide = env.collide()
        
        if is_collide or is_reached:
            if is_reached and not is_collide:
                consecutive_successes += 1
                print(f"[{speed_multiplier}x] Lap {consecutive_successes}/{required_laps} SUCCESS! (steps: {lap_steps})", flush=True)
            else:
                print(f"[{speed_multiplier}x] Collision! Resetting count from {consecutive_successes} to 0", flush=True)
                consecutive_successes = 0
            lap_steps = 0
            env.reset()
            env.sim_speed = speed_multiplier
            
    proc.stdin.close()
    proc.wait()
    print(f"[{speed_multiplier}x] Raw recording finished ({total_frames} frames). Converting to high-res GIF...", flush=True)
    
    if speed_multiplier == 1:
        # 1x: 71s duration. 720x360 resolution, 11 fps, 80 colors -> crisp, smooth, ~19MB
        cmd_gif = [
            ffmpeg_exe, '-y',
            '-i', raw_mp4,
            '-vf', 'fps=11,scale=720:360:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=80:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3',
            out_gif
        ]
    else:
        # 2x: 35.5s duration. 760x380 resolution, 14 fps, 96 colors -> high-res, smooth 14fps, ~17MB
        cmd_gif = [
            ffmpeg_exe, '-y',
            '-i', raw_mp4,
            '-vf', 'fps=14,scale=760:380:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3',
            out_gif
        ]
        
    subprocess.run(cmd_gif, check=True)
    if os.path.exists(raw_mp4):
        os.remove(raw_mp4)
        
    gif_sz = os.path.getsize(out_gif) / (1024 * 1024)
    print(f"[{speed_multiplier}x] Finished! {out_gif} size: {gif_sz:.1f}MB", flush=True)

if __name__ == "__main__":
    record_hires_run(1, seed=42, required_laps=3, out_gif="images/simulation_1x.gif")
    record_hires_run(2, seed=42, required_laps=3, out_gif="images/simulation_2x.gif")
