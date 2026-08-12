import pygame
import numpy as np
import math
import random
import datetime
import os
from scipy.spatial import cKDTree


# -------------------------------------------------------------
# 가장 빠른 속도를 위한 완전 벡터화 LiDAR + KD-tree
# -------------------------------------------------------------
def lidar_vectorized_np(boat_pos, boat_heading, rel_angles, obstacles, lidar_range, tree, query_rad=500):
    n_beams = rel_angles.shape[0]
    out = np.full(n_beams, lidar_range, dtype=np.float32)

    # 주변 장애물만 검사
    idx = tree.query_ball_point(boat_pos, query_rad)
    if len(idx) == 0:
        return out

    obs = obstacles[idx]
    ox = obs[:, 0]
    oy = obs[:, 1]
    orad = obs[:, 2]

    # 빔 방향 전체 계산
    angs = boat_heading + rel_angles
    cosA = np.cos(angs)
    sinA = np.sin(angs)

    dx = ox[None, :] - boat_pos[0]
    dy = oy[None, :] - boat_pos[1]

    proj = dx * cosA[:, None] + dy * sinA[:, None]
    valid = (proj > 0) & (proj < lidar_range)

    if not np.any(valid):
        return out

    proj_valid = np.where(valid, proj, np.inf)
    px = boat_pos[0] + proj_valid * cosA[:, None]
    py = boat_pos[1] + proj_valid * sinA[:, None]

    dd = (px - ox[None, :])**2 + (py - oy[None, :])**2
    hit = dd <= (orad[None, :]**2)

    for i in range(n_beams):
        hh = np.where(hit[i], proj_valid[i], np.inf)
        if np.any(hh < np.inf):
            out[i] = np.min(hh)

    return out


# -------------------------------------------------------------
# Heading 계산 개선판 (Goal + Repulsion + Gap)
# -------------------------------------------------------------
def compute_safe_heading(
    boat_heading, gps_heading, lidar, rel_cos, rel_sin,
    lidar_range, prev_gap_dir,
    k_goal=12.0, k_rep=2.0, k_gap=7.0, d0=300.0
):
    A = np.array([math.cos(gps_heading), math.sin(gps_heading)]) * (k_goal * 0.5)

    d = lidar
    rep_mag = np.clip((lidar_range*1.4 - d), 0, None) / (lidar_range*1.4)
    rep_mag = (rep_mag ** 2.2) * (k_rep * 2.8)

    rx_body = rep_mag * (-rel_cos)
    ry_body = rep_mag * (-rel_sin)

    cosH = math.cos(boat_heading)
    sinH = math.sin(boat_heading)

    R = np.zeros(2, dtype=np.float32)
    R[0] = np.sum(rx_body * cosH - ry_body * sinH)
    R[1] = np.sum(rx_body * sinH + ry_body * cosH)

    n = len(lidar)
    Lm = np.mean(lidar[:n//2])
    Rm = np.mean(lidar[n//2:])
    diff = (Rm - Lm) / lidar_range

    Le = np.mean(lidar[:8])
    Re = np.mean(lidar[-8:])
    diff2 = (Re - Le) / lidar_range

    combined = 0.6 * diff + 0.4 * diff2

    if abs(combined) < 0.12:
        gap_dir = prev_gap_dir
    else:
        gap_dir = 1 if combined > 0 else -1

    G = np.array([
        math.cos(boat_heading + gap_dir * 1.0),
        math.sin(boat_heading + gap_dir * 1.0)
    ]) * (k_gap * 1.9)

    V = A + R + G
    ang = math.atan2(V[1], V[0])
    return ang, gap_dir



# -------------------------------------------------------------
# Boat 환경 클래스
# -------------------------------------------------------------
class BoatEnv:
    def __init__(self, map_w=2000, map_h=600, lidar_beams=90, lidar_range=450, dt=0.04):
        pygame.init()
        pygame.font.init()

        self.map_w = map_w
        self.map_h = map_h
        self.screen = pygame.display.set_mode((map_w, map_h))
        pygame.display.set_caption("Boat Simulator")

        self.clock = pygame.time.Clock()
        self.dt = dt

        self.font = pygame.font.SysFont("consolas", 20)

        self.lidar_beams = lidar_beams
        self.lidar_range = int(lidar_range * 0.9)

        self.rel_angles = np.linspace(-np.pi/2, np.pi/2, lidar_beams).astype(np.float32)
        self.rel_cos = np.cos(self.rel_angles).astype(np.float32)
        self.rel_sin = np.sin(self.rel_angles).astype(np.float32)

        self.mass = 20.0
        self.inertia = 0.08
        self.drag = 0.40
        self.rot_drag = 0.60
        self.boat_radius = 25

        self.obs_count = 110
        self.obs_radius = 12
        self.min_obs_dist = 120

        self.trail_surface = pygame.Surface((map_w, map_h), pygame.SRCALPHA)
        self.frame_count = 0

        self.prev_steer = 0.0
        self.prev_gap_dir = 0
        self.last_desired = 0
        self.heading_smooth = 0
        self.smooth_alpha = 0.35

        self.last_pwm_L = 1500
        self.last_pwm_R = 1500

        self.reset()

    def reset(self):
        self.boat_pos = np.array([70.0, self.map_h / 2], dtype=np.float32)
        self.boat_vel = np.zeros(2, dtype=np.float32)
        self.boat_ang_vel = 0

        self.target = np.array([self.map_w - 200, self.map_h / 2], dtype=np.float32)
        self.trail_surface.fill((0, 0, 0, 0))

        obs = []
        attempts = 0
        while len(obs) < self.obs_count and attempts < 10000:
            attempts += 1
            x = random.randint(5, self.map_w - 5)
            y = random.randint(5, self.map_h - 5)

            pos = np.array([x, y])

            if np.linalg.norm(pos - self.target) < 180: continue
            if np.linalg.norm(pos - self.boat_pos) < 180: continue

            if all(np.linalg.norm(pos - np.array([ox, oy])) >= self.min_obs_dist for ox, oy, _ in obs):
                obs.append((x, y, self.obs_radius))

        self.obstacles = np.array(obs, dtype=np.float32)
        self.tree = cKDTree(self.obstacles[:, :2])

        dx = self.target[0] - self.boat_pos[0]
        dy = self.target[1] - self.boat_pos[1]
        self.boat_heading = math.atan2(dy, dx)

        self.last_lidar = np.zeros(self.lidar_beams, dtype=np.float32)
        self.prev_gap_dir = 0
        self.heading_smooth = self.boat_heading

    def pwm_to_thrust(self, pwm):
        return pwm * 3.0

    def step(self, pwm_L, pwm_R):
        self.last_pwm_L = pwm_L
        self.last_pwm_R = pwm_R

        tL = self.pwm_to_thrust(pwm_L)
        tR = self.pwm_to_thrust(pwm_R)

        forward = tL + tR
        moment = (tR - tL) * 0.006

        hv = np.array([
            math.cos(self.boat_heading),
            math.sin(self.boat_heading)
        ])

        acc = forward / self.mass
        drag = -self.drag * np.linalg.norm(self.boat_vel) * self.boat_vel

        self.boat_vel += (acc * hv + drag) * self.dt
        prev = self.boat_pos.copy()
        self.boat_pos += self.boat_vel * self.dt

        if self.frame_count % 2 == 0:
            pygame.draw.line(
                self.trail_surface, (0, 120, 255, 255),
                (int(prev[0]), int(prev[1])),
                (int(self.boat_pos[0]), int(self.boat_pos[1])),
                2
            )

        ang_acc = (moment - self.rot_drag * self.boat_ang_vel) / self.inertia
        self.boat_ang_vel += ang_acc * self.dt
        self.boat_ang_vel = np.clip(self.boat_ang_vel, -1.2, 1.2)
        self.boat_heading += self.boat_ang_vel * self.dt

    def lidar_scan(self):
        if self.frame_count % 2 == 0:
            self.last_lidar = lidar_vectorized_np(
                self.boat_pos,
                self.boat_heading,
                self.rel_angles,
                self.obstacles,
                self.lidar_range,
                self.tree
            )
        return self.last_lidar

    def collide(self):
        ox = self.obstacles[:, 0]
        oy = self.obstacles[:, 1]
        r = self.obstacles[:, 2] + self.boat_radius

        dx = ox - self.boat_pos[0]
        dy = oy - self.boat_pos[1]

        hit = np.any(dx*dx + dy*dy <= r*r)

        wall = (
            self.boat_pos[0] <= 0 or
            self.boat_pos[0] >= self.map_w or
            self.boat_pos[1] <= 0 or
            self.boat_pos[1] >= self.map_h
        )
        return hit or wall

    def heading_control(self, desired, current, ang_vel):
        err = (desired - current + np.pi) % (2 * np.pi) - np.pi
        steer = 1.35 * err - 0.28 * ang_vel
        steer = np.clip(steer, -1, 1)
        steer = 0.45 * steer + 0.55 * self.prev_steer
        self.prev_steer = steer
        return steer

    def get_pwm_pair(self, steer):
        PWM_center = 1500
        PWM_range = 210

        mag = abs(steer)
        m = np.log1p(4 * mag) / np.log(5)
        delta = m * PWM_range

        if steer >= 0:
            pwm_L = PWM_center - delta
            pwm_R = PWM_center + delta
        else:
            pwm_L = PWM_center + delta
            pwm_R = PWM_center - delta

        return int(np.clip(pwm_L, 1300, 1700)), int(np.clip(pwm_R, 1300, 1700))

    def draw_text(self, txt, x, y):
        o = self.font.render(txt, True, (0, 0, 0))
        self.screen.blit(o, (x - 1, y - 1))
        self.screen.blit(o, (x + 1, y - 1))
        self.screen.blit(o, (x - 1, y + 1))
        self.screen.blit(o, (x + 1, y + 1))
        self.screen.blit(self.font.render(txt, True, (255, 255, 255)), (x, y))

    def render(self):
        self.screen.fill((235, 235, 235))
        self.screen.blit(self.trail_surface, (0, 0))

        for ox, oy, r in self.obstacles:
            pygame.draw.circle(self.screen, (80, 80, 80), (int(ox), int(oy)), int(r))

        pygame.draw.circle(self.screen, (255, 0, 0), (int(self.target[0]), int(self.target[1])), 26)

        bx, by = self.boat_pos
        h = self.boat_heading
        ch = math.cos(h)
        sh = math.sin(h)

        GAP = 10
        L = 80
        W = 15

        left = (bx - sh * GAP, by + ch * GAP)
        right = (bx + sh * GAP, by - ch * GAP)

        hull = [
            (L * 0.50, 0),
            (L * 0.10, W),
            (-L * 0.30, W * 0.8),
            (-L * 0.48, W * 0.55),
            (-L * 0.50, 0),
            (-L * 0.48, -W * 0.55),
            (-L * 0.30, -W * 0.8),
            (L * 0.10, -W),
        ]

        def TR(c, px, py):
            return (int(c[0] + px * ch - py * sh), int(c[1] + px * sh + py * ch))

        left_hull = [TR(left, p[0], p[1]) for p in hull]
        right_hull = [TR(right, p[0], p[1]) for p in hull]

        pygame.draw.polygon(self.screen, (0, 0, 240), left_hull)
        pygame.draw.polygon(self.screen, (0, 0, 240), right_hull)

        deg_des = math.degrees(self.last_desired)
        deg_cur = math.degrees(self.boat_heading)

        sx, sy = 20, 20
        g = 25

        self.draw_text(f"Desired Heading : {deg_des:.1f}", sx, sy); sy += g
        self.draw_text(f"Current Heading : {deg_cur:.1f}", sx, sy); sy += g
        self.draw_text(f"PWM Left       : {self.last_pwm_L}", sx, sy); sy += g
        self.draw_text(f"PWM Right      : {self.last_pwm_R}", sx, sy); sy += g

        pygame.display.update()
        self.clock.tick(60)

    def increment_frame(self):
        self.frame_count += 1

    def save_screenshot(self):
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = f"screenshot_{now}.png"
        pygame.image.save(self.screen, fname)
        print("[Saved]", fname)



# -------------------------------------------------------------
# 메인 루프
# -------------------------------------------------------------
def run():
    env = BoatEnv()

    while True:
        env.increment_frame()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                return

        lidar = env.lidar_scan()

        dx = env.target[0] - env.boat_pos[0]
        dy = env.target[1] - env.boat_pos[1]
        gps_heading = math.atan2(dy, dx)

        desired, new_gap = compute_safe_heading(
            env.boat_heading, gps_heading, lidar,
            env.rel_cos, env.rel_sin,
            env.lidar_range, env.prev_gap_dir
        )

        env.prev_gap_dir = new_gap
        env.last_desired = desired

        env.heading_smooth = (
            env.smooth_alpha * desired +
            (1 - env.smooth_alpha) * env.heading_smooth
        )

        steer = env.heading_control(env.heading_smooth, env.boat_heading, env.boat_ang_vel)
        pwm_L, pwm_R = env.get_pwm_pair(steer)

        env.step(pwm_L, pwm_R)
        env.render()

        if env.collide() or np.linalg.norm(env.target - env.boat_pos) < 70:
            env.save_screenshot()
            env.reset()


if __name__ == "__main__":
    run()