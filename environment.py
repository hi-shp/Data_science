import pygame
import numpy as np
import math
import random
from config import WIDTH, HEIGHT, GRID, GRID_W, GRID_H
from utils import wrap
from perception import init_grid
from navigation import reactive_avoidance
from ui_renderer import EnvRenderer

class BoatEnv:
    def __init__(self):
        pygame.init()
        self.w = WIDTH
        self.h = HEIGHT
        self.sim_h = 630
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("kaboat simulation")
        self.clock = pygame.time.Clock()
        self.dt = 0.04
         
        self.lidar_beams = 180
        self.lidar_range = 350
        self.rel_angles = np.linspace(-np.pi, np.pi, self.lidar_beams, endpoint=False)
        
        self.mass = 20
        self.inertia = 6
        self.drag = 0.38
        self.rot_drag = 0.55
        self.boat_radius = 25
        
        self.trail = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        self.path_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        self.wake_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        self.occ_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        self.shadow_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        
        self.obs_n = 80
        self.obs_r = 17
        self.min_obs = 120
        
        self.grid = init_grid()
        self.clusters = []
        self.cluster_ids = []
        self.current_wp = None
        self.next_wp = None
        self.visited = set()
        
        self.frame = 0
        self.prev_steer = 0
        self.wp_check_timer = 0
        self.steer_timer = 0
        self.path_timer = 0
        
        self.bezier_path = None
        self.next_bezier_path = None
        self.pursuit_target = None
        self.next_pursuit_target = None
        self.wakes = [] # [x, y, radius, alpha, type, life, max_life, angle_offset]
        
        self.obstacles = np.array([])
        self.dynamic_obstacles = np.array([])
        
        self.show_path1 = True
        self.show_path2 = True
        self.show_lidar = True
        self.show_lidar_range = True
        
        self.cb1_rect = pygame.Rect(40, 670, 20, 20)
        self.cb2_rect = pygame.Rect(40, 710, 20, 20)
        self.cb3_rect = pygame.Rect(40, 750, 20, 20)
        self.cb4_rect = pygame.Rect(40, 790, 20, 20)
        
        self.renderer = EnvRenderer(self)
        self.reset()

    def reset(self):
        self.boat_pos = np.array([65, self.sim_h/2], dtype=np.float32)
        self.boat_vel = np.zeros(2)
        self.boat_ang_vel = 0
        self.target = np.array([self.w - 100, self.sim_h/2], dtype=np.float32)
        
        self.trail.fill((0, 0, 0, 0))
        self.path_surf.fill((0, 0, 0, 0))
        self.wake_surf.fill((0, 0, 0, 0))
        
        obs = []
        t = 0
        while len(obs) < self.obs_n and t < 5000:
            t += 1
            x = random.randint(300, self.w - 300)
            y = random.randint(30, self.sim_h - 30)
            p = np.array([x, y])
            if np.linalg.norm(p - self.target) < 180: continue
            if np.linalg.norm(p - self.boat_pos) < 180: continue
            
            ok = True
            for (ox, oy, r) in obs:
                if np.linalg.norm(p - np.array([ox, oy])) < self.min_obs:
                    ok = False
                    break
            if ok:
                obs.append((x, y, self.obs_r))
                
        self.obstacles = np.array(obs, dtype=np.float32)
        self.dynamic_obstacles = self.obstacles.copy()
        
        dx = self.target[0] - self.boat_pos[0]
        dy = self.target[1] - self.boat_pos[1]
        self.boat_heading = math.atan2(dy, dx)
        
        self.grid[:] = 0
        self.clusters = []
        self.cluster_ids = []
        self.current_wp = None
        self.next_wp = None
        self.visited = set()
        
        self.wp_check_timer = 0
        self.steer_timer = 0
        self.path_timer = 0
        self.bezier_path = None
        self.next_bezier_path = None
        self.pursuit_target = None
        self.next_pursuit_target = None
        self.wakes = []
        self.emergency_mode = False

    def handle_click(self, pos):
        if self.cb1_rect.collidepoint(pos):
            self.show_path1 = not self.show_path1
        elif self.cb2_rect.collidepoint(pos):
            self.show_path2 = not self.show_path2
        elif self.cb3_rect.collidepoint(pos):
            self.show_lidar = not self.show_lidar
        elif self.cb4_rect.collidepoint(pos):
            self.show_lidar_range = not self.show_lidar_range

    def update_dynamic_obstacles(self):
        self.dynamic_obstacles = self.obstacles.copy()
        for i in range(len(self.obstacles)):
            ox, oy, r = self.obstacles[i]
            sway_x = math.sin(self.frame * 0.03 + oy * 0.1) * (r * 0.2)
            sway_y = math.cos(self.frame * 0.04 + ox * 0.1) * (r * 0.2)
            self.dynamic_obstacles[i, 0] = ox + sway_x
            self.dynamic_obstacles[i, 1] = oy + sway_y

    def pwm_to_thrust(self, p):
        return p * 10

    def step(self, L, R):
        tL = self.pwm_to_thrust(L)
        tR = self.pwm_to_thrust(R)
        target_fwd = (tL + tR) / 9.0
        
        if self.emergency_mode:
            target_fwd = 0.0
            
        if not hasattr(self, 'current_fwd'):
            self.current_fwd = 0.0
            
        self.current_fwd = self.current_fwd * 0.95 + target_fwd * 0.05
        mom = (tR - tL) * 0.006
        hv = np.array([math.cos(self.boat_heading), math.sin(self.boat_heading)])
        
        acc = self.current_fwd / self.mass
        vel_norm = np.linalg.norm(self.boat_vel)
        if vel_norm > 0:
            drag = -self.drag * vel_norm * self.boat_vel
        else:
            drag = np.zeros(2)
            
        prev = self.boat_pos.copy()
        self.boat_vel += (acc * hv + drag) * self.dt
        self.boat_pos += self.boat_vel * self.dt
        
        if self.frame % 7 == 0:
            pygame.draw.line(self.trail, (255, 255, 255, 60),
                             (int(prev[0]), int(prev[1])),
                             (int(self.boat_pos[0]), int(self.boat_pos[1])), 2)
                             
        ang_acc = (mom - self.rot_drag * self.boat_ang_vel) / self.inertia
        self.boat_ang_vel += ang_acc * self.dt
        self.boat_ang_vel *= 0.84
        self.boat_heading += self.boat_ang_vel * self.dt

        # 디테일한 파도 생성 로직
        if vel_norm > 3:
            h = self.boat_heading
            intensity = min(1.0, vel_norm / 15.0) # 속도에 따른 파도 강도
            
            # 1. 중앙 프로펠러 난류 (자주 발생, 짧게 유지)
            if self.frame % 2 == 0:
                px = self.boat_pos[0] - math.cos(h) * 25
                py = self.boat_pos[1] - math.sin(h) * 25
                # x, y, 반경, 알파, 타입(0=중앙), 현재수명, 최대수명, 각도오프셋
                self.wakes.append([px + random.uniform(-3, 3), py + random.uniform(-3, 3), 
                                   1.0 + intensity, 180 * intensity, 0, 0, 40, 0])

            # 2. 좌우 V자 주 파도 (Kelvin Wake)
            if self.frame % 4 == 0:
                spread_angle = 0.55 # 파도가 퍼지는 기본 각도
                
                # 좌측 주 파도
                lx = self.boat_pos[0] - math.cos(h + 0.3) * 15
                ly = self.boat_pos[1] - math.sin(h + 0.3) * 15
                self.wakes.append([lx, ly, 1.5, 140 * intensity, 1, 0, 100, spread_angle])
                
                # 우측 주 파도
                rx = self.boat_pos[0] - math.cos(h - 0.3) * 15
                ry = self.boat_pos[1] - math.sin(h - 0.3) * 15
                self.wakes.append([rx, ry, 1.5, 140 * intensity, -1, 0, 100, -spread_angle])

            # 3. 바깥쪽 잔물결 (속도가 높을 때만 추가 생성)
            if vel_norm > 8 and self.frame % 6 == 0:
                outer_angle = 0.8
                lx_o = self.boat_pos[0] - math.cos(h + 0.5) * 10
                ly_o = self.boat_pos[1] - math.sin(h + 0.5) * 10
                self.wakes.append([lx_o, ly_o, 1.0, 90 * intensity, 2, 0, 80, outer_angle])
                
                rx_o = self.boat_pos[0] - math.cos(h - 0.5) * 10
                ry_o = self.boat_pos[1] - math.sin(h - 0.5) * 10
                self.wakes.append([rx_o, ry_o, 1.0, 90 * intensity, -2, 0, 80, -outer_angle])

    def collide(self):
        ox = self.dynamic_obstacles[:, 0]
        oy = self.dynamic_obstacles[:, 1]
        rr = self.dynamic_obstacles[:, 2] + self.boat_radius
        dx = ox - self.boat_pos[0]
        dy = oy - self.boat_pos[1]
        hit = np.any(dx*dx + dy*dy <= rr*rr)
        wall = (self.boat_pos[0] <= 0 or self.boat_pos[0] >= self.w or
                self.boat_pos[1] <= 0 or self.boat_pos[1] >= self.sim_h)
        return hit or wall

    def get_pwm(self, steer):
        dead = 0.03
        if abs(steer) < dead: steer = 0
        mid = 1500; rng = 210
        m = np.log1p(4 * abs(steer)) / np.log(5)
        d = m * rng
        if steer >= 0: L = mid - d; R = mid + d
        else: L = mid + d; R = mid - d
        return int(np.clip(L, 1300, 1700)), int(np.clip(R, 1300, 1700))

    def validate_wp_grid(self):
        if self.current_wp is None: return
        self.wp_check_timer += self.dt
        if self.wp_check_timer < 0.05: return
        self.wp_check_timer = 0
        wp = self.current_wp["pos"]; pair = self.current_wp["pair"]
        gx = int(wp[0] // GRID); gy = int(wp[1] // GRID); rad = int(35 // GRID)
        for yy in range(max(0, gy - rad), min(GRID_H, gy + rad + 1)):
            for xx in range(max(0, gx - rad), min(GRID_W, gx + rad + 1)):
                if self.grid[yy, xx] >= 3:
                    self.visited.add(pair); self.visited.add((pair[1], pair[0]))
                    self.current_wp = None; return

    def validate_wp_obstacle_5x5(self):
        if self.current_wp is None: return
        wp = self.current_wp["pos"]
        gx = int(wp[0] // GRID); gy = int(wp[1] // GRID)
        xs = range(gx - 2, gx + 3); ys = range(gy - 2, gy + 3)
        ox = self.dynamic_obstacles[:, 0]; oy = self.dynamic_obstacles[:, 1]; rr = self.dynamic_obstacles[:, 2]
        for yy in ys:
            for xx in xs:
                if 0 <= xx < GRID_W and 0 <= yy < GRID_H:
                    cx = xx * GRID + GRID * 0.5; cy = yy * GRID + GRID * 0.5
                    dx = ox - cx; dy = oy - cy
                    hit = np.any(dx*dx + dy*dy <= rr*rr)
                    if hit:
                        p = self.current_wp["pair"]
                        self.visited.add(p); self.visited.add((p[1], p[0]))
                        self.current_wp = None; return

    def update_steering(self, dists):
        self.steer_timer += self.dt
        if self.steer_timer < 0.02: return None
        self.steer_timer = 0
        center_idx = self.lidar_beams // 2
        span = self.lidar_beams // 12
        front_dists = dists[center_idx - span : center_idx + span]
        min_front_dist = np.min(front_dists)
        if min_front_dist < 70: self.emergency_mode = True; self.current_wp = None
        else: self.emergency_mode = False
        if self.pursuit_target is None: return 0
        px, py = self.pursuit_target
        heading_target = math.atan2(py - self.boat_pos[1], px - self.boat_pos[0])
        heading_error = wrap(heading_target - self.boat_heading)
        steer_raw = heading_error * 0.6
        steer_f = 0.3 * steer_raw + 0.7 * self.prev_steer
        self.prev_steer = steer_f
        avoid = reactive_avoidance(dists, self.rel_angles)
        avoid_multiplier = 0.1 if self.emergency_mode else 0.02
        return np.clip(steer_f + avoid_multiplier * avoid, -1, 1)

    def render(self, hits):
        self.renderer.render(hits)