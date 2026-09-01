import os
import json
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

        # 학습된 최적 파라미터 자동 로드
        self.params = {
            'steer_gain': 0.7752,
            'steer_alpha': 0.3515,
            'mom_coeff': 0.00665,
            'pwm_rng': 270.36,
            'avoid_normal': 0.019,
            'avoid_em': 0.11,
            'clear_margin': 2.15,
            'em_enter': 115.0,
            'em_exit': 160.0,
            'em_hold_frames': 18,
            'align_exp': 5.0,
            'fwd_exp': 3.0,
            'clear_exp': 1.5,
            'width_exp': 0.2,
            'cluster_pen_w': 0.5,
            'wp_switch_thresh': 1.15
        }
        json_path = "best_learned_params.json"
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    self.params.update(json.load(f))
            except Exception:
                pass
         
        self.lidar_beams = 180
        self.lidar_range = 350
        self.rel_angles = np.linspace(-np.pi, np.pi, self.lidar_beams, endpoint=False)
        
        self.mass = 12
        self.inertia = 6
        self.drag = 0.2
        self.rot_drag = 1.3
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
        self.wakes = [] # [x, y, radius, alpha]
        self.reflected_wakes = [] # [x, y, radius, alpha] (장애물 충돌 반사파)
        
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
        
        self.sim_speed = 1
        self.speed_btns = {
            1: pygame.Rect(40, 835, 52, 28),
            2: pygame.Rect(100, 835, 52, 28),
            3: pygame.Rect(160, 835, 52, 28),
            4: pygame.Rect(220, 835, 52, 28)
        }
        
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
        else:
            for spd, rect in self.speed_btns.items():
                if rect.collidepoint(pos):
                    self.sim_speed = spd
                    break

    def update_dynamic_obstacles(self):
        self.dynamic_obstacles = self.obstacles.copy()
        for i in range(len(self.obstacles)):
            ox, oy, r = self.obstacles[i]
            phase = self.frame * 0.04 + ox * 0.05 + oy * 0.05
            sway_x = math.sin(phase) * (r * 0.2)
            sway_y = math.cos(phase * 1.2) * (r * 0.2)
            self.dynamic_obstacles[i, 0] = ox + sway_x
            self.dynamic_obstacles[i, 1] = oy + sway_y
            
            # 부표 중앙을 기준으로 부드러운 백색 원형 구름 파도가 더 자주 잔잔하게 퍼져나감
            if (self.frame + i * 19) % 36 == 0:
                self.reflected_wakes.append([
                    self.dynamic_obstacles[i, 0], self.dynamic_obstacles[i, 1], r + 1.0, 72
                ])

    def pwm_to_thrust(self, p):
        return p * 10

    def step(self, L, R):
        tL = self.pwm_to_thrust(L)
        tR = self.pwm_to_thrust(R)
        # 220도 범위 내 최소 장애물 거리에 비례하여 연속적으로 속도 조절
        em_dist = getattr(self, 'min_wide_dist', 999)
        if em_dist < 200:
            # 거리 0px -> /20(최대감속), 거리 200px -> /9(정상속도)
            ratio = 9.0 + (200.0 - em_dist) / 200.0 * 11.0
            target_fwd = (tL + tR) / ratio
        else:
            target_fwd = (tL + tR) / 9.0
            
        if not hasattr(self, 'current_fwd'):
            self.current_fwd = 0.0
            
        self.current_fwd = self.current_fwd * 0.95 + target_fwd * 0.05
        mom = (tR - tL) * self.params['mom_coeff']
        hv = np.array([math.cos(self.boat_heading), math.sin(self.boat_heading)])
        
        acc = self.current_fwd / self.mass
        vel_norm = np.linalg.norm(self.boat_vel)
        
        # 유체 항력
        drag = -self.drag * self.boat_vel * vel_norm
        
        # 횡방향 슬립 댐핑
        lat_v = np.array([-math.sin(self.boat_heading), math.cos(self.boat_heading)])
        lat_speed = np.dot(self.boat_vel, lat_v)
        drag += -lat_v * lat_speed * 18.0
            
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
        
        d_head = self.boat_ang_vel * self.dt
        self.boat_heading += d_head
        
        # 선미 추진 선박의 후방 회전축(L_pivot = 4.0px)에 따른 자연스러운 선회 궤적
        L_pivot = 4.0
        lat_vec = np.array([-math.sin(self.boat_heading), math.cos(self.boat_heading)])
        self.boat_pos += lat_vec * (self.boat_ang_vel * L_pivot * self.dt)

        # 실제 선박 유체역학 파도 생성 (Realistic Hydrodynamic Wave System)
        if vel_norm > 2.0:
            h = self.boat_heading
            intensity = min(1.0, vel_norm / 11.0)
            sh = math.sin(h); ch = math.cos(h)
            GAP = 11; L = 84

            # 선미 듀얼 쓰러스터 추진 제트 기포 및 후방 횡단 웨이크 (Enlarged Stern Roostertail & Trailing Foam)
            if self.frame % 2 == 0:
                stern_lx = self.boat_pos[0] - sh * GAP - ch * (L * 0.50)
                stern_ly = self.boat_pos[1] + ch * GAP - sh * (L * 0.50)
                stern_rx = self.boat_pos[0] + sh * GAP - ch * (L * 0.50)
                stern_ry = self.boat_pos[1] - ch * GAP - sh * (L * 0.50)
                
                self.wakes.append([stern_lx + random.uniform(-1.5, 1.5), stern_ly + random.uniform(-1.5, 1.5), 3.0, 180 * intensity, -ch * 0.65, -sh * 0.65])
                self.wakes.append([stern_rx + random.uniform(-1.5, 1.5), stern_ry + random.uniform(-1.5, 1.5), 3.0, 180 * intensity, -ch * 0.65, -sh * 0.65])
                
            if self.frame % 3 == 0:
                cx = self.boat_pos[0] - ch * 42
                cy = self.boat_pos[1] - sh * 42
                self.wakes.append([cx + random.uniform(-2.5, 2.5), cy + random.uniform(-2.5, 2.5), 4.5, 130 * intensity, -ch * 0.85, -sh * 0.85])

            # 좌/우 회전 시 외측 선체 유체 저항에 의한 흰색 거품 (Outer Hull Resistance Foam)
            if abs(self.boat_ang_vel) > 0.06:
                turn_p = min(1.0, abs(self.boat_ang_vel) / 0.42) * intensity
                s = 1.0 if self.boat_ang_vel < 0 else -1.0
                
                rand_l = random.uniform(-L * 0.25, L * 0.15)
                bx_foam = self.boat_pos[0] + s * (-sh) * (GAP + random.uniform(1.5, 4.0)) + ch * rand_l
                by_foam = self.boat_pos[1] + s * ch * (GAP + random.uniform(1.5, 4.0)) + sh * rand_l
                
                drift_vx = s * (-sh) * random.uniform(0.3, 0.7) - ch * 0.35
                drift_vy = s * ch * random.uniform(0.3, 0.7) - sh * 0.35
                init_r = random.uniform(2.0, 3.5)
                alpha = random.uniform(140, 200) * turn_p
                
                # 7번째 원소=1: 순백색 거품 태그 (뷰쪽 파란색 링 없이 흰색만)
                self.wakes.append([bx_foam, by_foam, init_r, alpha, drift_vx, drift_vy, 1])

        # 파도-장애물 물리 상호작용 (Wave Absorption & Frothy Micro-Bubble Scattering)
        if len(self.wakes) > 0 and len(self.dynamic_obstacles) > 0:
            for w in self.wakes:
                for ox, oy, ob_r in self.dynamic_obstacles:
                    dx = w[0] - ox; dy = w[1] - oy
                    dist = math.hypot(dx, dy)
                    
                    # 1. 장애물 내부로 들어간 파도는 완전히 소멸/흡수 (Absorption)
                    if dist < ob_r + 2:
                        w[3] = 0
                        break
                    
                    # 2. 장애물 둘레에 파도가 닿으면 자글자글한 초미세 나노 거품 파편들이 반사 산란 (Nano-Spray Droplets)
                    if abs(dist - (w[2] + ob_r)) < 5.0 and w[3] > 35:
                        if random.random() < 0.35:
                            for _ in range(random.randint(2, 4)):
                                angle = math.atan2(dy, dx) + random.uniform(-0.8, 0.8)
                                spd = random.uniform(0.8, 1.8)
                                fx = ox + math.cos(angle) * (ob_r + random.uniform(0.8, 2.2))
                                fy = oy + math.sin(angle) * (ob_r + random.uniform(0.8, 2.2))
                                self.reflected_wakes.append([
                                    fx, fy, random.uniform(0.3, 0.65), w[3] * 0.85,
                                    math.cos(angle) * spd, math.sin(angle) * spd
                                ])

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
        dead = 0.02
        if abs(steer) < dead: steer = 0
        mid = 1500; rng = self.params['pwm_rng']
        m = np.log1p(3 * abs(steer)) / np.log(4)
        d = m * rng
        if steer >= 0: L = mid - d; R = mid + d
        else: L = mid + d; R = mid - d
        return int(np.clip(L, 1230, 1770)), int(np.clip(R, 1230, 1770))

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
        center_idx = self.lidar_beams // 2
        # 정면 + 양옆 20도 = 총 220도 범위 감시
        span = int(self.lidar_beams * 220 / 360 / 2)
        front_dists = dists[center_idx - span : center_idx + span]
        min_front_dist = np.min(front_dists)
        self.min_wide_dist = min_front_dist
        
        if not hasattr(self, 'emergency_cooldown'):
            self.emergency_cooldown = 0
            
        if min_front_dist < self.params['em_enter']:
            self.emergency_mode = True
            self.emergency_cooldown = self.params['em_hold_frames']
        elif self.emergency_mode:
            self.emergency_cooldown -= 1
            if min_front_dist > self.params['em_exit'] and self.emergency_cooldown <= 0:
                self.emergency_mode = False

        if self.pursuit_target is None: return 0
        px, py = self.pursuit_target
        heading_target = math.atan2(py - self.boat_pos[1], px - self.boat_pos[0])
        heading_error = wrap(heading_target - self.boat_heading)

        if self.emergency_mode:
            steer_gain = 0.95
            avoid_multiplier = self.params['avoid_em']
        else:
            steer_gain = self.params['steer_gain']
            avoid_multiplier = self.params['avoid_normal']
            
        # 각속도 댐핑으로 관성 오버슈트 억제
        d_term = -0.15 * getattr(self, 'boat_ang_vel', 0.0)
        steer_raw = heading_error * steer_gain + d_term
        alpha = self.params['steer_alpha']
        steer_f = alpha * steer_raw + (1.0 - alpha) * self.prev_steer
        self.prev_steer = steer_f
        
        avoid = reactive_avoidance(dists, self.rel_angles)
        
        # 반발력 정상 작동: 웨이포인트 선회 방향과 반대로 충돌할 때만 상쇄 방지를 위해 소프트 감쇠(0.25) 적용
        if self.current_wp is not None and (steer_f * avoid < 0) and abs(steer_f) > 0.15:
            avoid *= 0.25
            
        return np.clip(steer_f + avoid_multiplier * avoid, -1, 1)

    def render(self, hits):
        self.renderer.render(hits)