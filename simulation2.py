import pygame
import numpy as np
import math
import random
import datetime
import os

class BoatEnv:
    def __init__(self,
                 map_w=2000,
                 map_h=600,
                 lidar_beams=90,
                 lidar_range=450,
                 dt=0.04):

        pygame.init()
        pygame.font.init()

        self.map_w = map_w
        self.map_h = map_h
        self.screen = pygame.display.set_mode((map_w, map_h))
        pygame.display.set_caption("Boat Simulator")

        self.clock = pygame.time.Clock()
        self.dt = dt

        try:
            self.font = pygame.font.SysFont("consolas", 20)
        except:
            self.font = pygame.font.Font(None, 20)

        self.lidar_beams = lidar_beams
        self.lidar_range = int(lidar_range * 0.8)

        self.rel_angles = np.linspace(-np.pi/2, np.pi/2, self.lidar_beams)
        self.rel_cos = np.cos(self.rel_angles)
        self.rel_sin = np.sin(self.rel_angles)

        self.mass = 20.0
        self.inertia = 0.08
        self.drag = 0.40
        self.rot_drag = 0.60

        self.boat_radius = 10

        self.obs_count = 100
        self.obs_radius = 10
        self.min_obs_dist = 90

        self.obstacles = []
        self.target = None

        self.trail_surface = pygame.Surface((map_w, map_h), pygame.SRCALPHA)
        self.frame_count = 0

        self.prev_steer = 0.0
        self.heading_hist = []
        self.heading_hist_len = 1

        self.prev_gap_dir = 0
        self.gap_threshold = 0.30

        self.last_pwm_L = 1500
        self.last_pwm_R = 1500
        self.last_desired = 0.0
        self.last_steer = 0.0

        self.reset()

    def reset(self):

        self.boat_pos = np.array([50.0, self.map_h / 2], dtype=np.float32)
        self.boat_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.boat_ang_vel = 0.0

        self.target = np.array([self.map_w - 200.0, self.map_h / 2], dtype=np.float32)

        self.trail_surface.fill((0,0,0,0))

        self.obstacles = []
        attempts = 0

        while len(self.obstacles) < self.obs_count and attempts < 8000:
            attempts += 1
            x = random.randint(200, self.map_w - 200)
            y = random.randint(120, self.map_h - 120)
            pos = np.array([x, y])

            if np.linalg.norm(pos - self.target) < 200:
                continue
            if np.linalg.norm(pos - self.boat_pos) < 200:
                continue

            ok = True
            for (ox, oy, r) in self.obstacles:
                if np.linalg.norm(pos - np.array([ox, oy])) < self.min_obs_dist:
                    ok = False
                    break

            if ok:
                self.obstacles.append((x, y, self.obs_radius))

        dx = self.target[0] - self.boat_pos[0]
        dy = self.target[1] - self.boat_pos[1]
        self.boat_heading = math.atan2(dy, dx)

        self.last_lidar = np.zeros(self.lidar_beams, dtype=np.float32)
        self.prev_steer = 0.0
        self.heading_hist = []
        self.prev_gap_dir = 0

    def pwm_to_thrust(self, pwm):
        return pwm * 3.0

    def step(self, pwm_L, pwm_R):

        self.last_pwm_L = pwm_L
        self.last_pwm_R = pwm_R

        px, py = self.boat_pos

        tL = self.pwm_to_thrust(pwm_L)
        tR = self.pwm_to_thrust(pwm_R)

        forward = tL + tR
        moment = (tR - tL) * 0.006

        hv = np.array([math.cos(self.boat_heading),
                       math.sin(self.boat_heading)])

        acc = forward / self.mass
        drag = -self.drag * np.linalg.norm(self.boat_vel) * self.boat_vel

        self.boat_vel += (acc * hv + drag) * self.dt
        self.boat_pos += self.boat_vel * self.dt

        if self.frame_count % 2 == 0:
            pygame.draw.line(
                self.trail_surface,
                (0,120,255,255),
                (int(px), int(py)),
                (int(self.boat_pos[0]), int(self.boat_pos[1])),
                2
            )

        ang_acc = (moment - self.rot_drag * self.boat_ang_vel) / self.inertia
        self.boat_ang_vel += ang_acc * self.dt
        self.boat_ang_vel = np.clip(self.boat_ang_vel, -1.2, 1.2)
        self.boat_heading += self.boat_ang_vel * self.dt

    def lidar_scan(self):
        if self.frame_count % 2 == 0:
            out = []
            for i in range(self.lidar_beams):
                ang = self.boat_heading + self.rel_angles[i]
                out.append(self.raycast(ang))
            self.last_lidar = np.array(out, dtype=np.float32)
        return self.last_lidar

    def raycast(self, angle):
        step_d = 5
        cosA = math.cos(angle)
        sinA = math.sin(angle)

        for d in range(0, self.lidar_range, step_d):
            x = self.boat_pos[0] + cosA * d
            y = self.boat_pos[1] + sinA * d

            if x < 0 or x >= self.map_w or y < 0 or y >= self.map_h:
                return d

            for ox, oy, r in self.obstacles:
                if abs(ox - x) > 80:
                    continue
                if abs(oy - y) > 80:
                    continue
                if (x - ox)**2 + (y - oy)**2 <= r*r:
                    return d

        return self.lidar_range

    def avoidance_strength(self, d):
        x = (self.lidar_range * 1.2 - d) / (self.lidar_range * 1.2)
        return x * 2.2

    def compute_safe_heading(self, gps_heading, lidar):

        cosH = math.cos(self.boat_heading)
        sinH = math.sin(self.boat_heading)

        A = np.array([
            math.cos(gps_heading),
            math.sin(gps_heading)
        ]) * 15.0

        R = np.zeros(2, dtype=np.float32)

        for i in range(self.lidar_beams):
            d = lidar[i]
            rep = self.avoidance_strength(d)

            rx = cosH * self.rel_cos[i] - sinH * self.rel_sin[i]
            ry = sinH * self.rel_cos[i] + cosH * self.rel_sin[i]

            R[0] -= rep * rx
            R[1] -= rep * ry

        left_gap  = np.mean(lidar[: self.lidar_beams//2])
        right_gap = np.mean(lidar[self.lidar_beams//2 :])

        score_L = left_gap  / self.lidar_range
        score_R = right_gap / self.lidar_range
        diff = score_R - score_L

        L_side = np.mean(lidar[:10]) / self.lidar_range
        R_side = np.mean(lidar[-10:]) / self.lidar_range
        diff2 = R_side - L_side

        combined = 0.7*diff + 0.3*diff2

        if abs(combined) < 0.30:
            gap_dir = self.prev_gap_dir
        else:
            gap_dir = 1 if combined > 0 else -1

        self.prev_gap_dir = gap_dir

        G = np.array([
            math.cos(self.boat_heading + gap_dir * 0.8),
            math.sin(self.boat_heading + gap_dir * 0.8)
        ]) * 7.0

        V = A + R + G
        ang = math.atan2(V[1], V[0])

        self.last_desired = ang
        return ang

    def heading_control(self, desired, current, ang_vel):
        err = (desired - current + np.pi) % (2*np.pi) - np.pi
        steer = 1.4 * err - 0.25 * ang_vel
        steer = np.clip(steer, -1, 1)
        steer = 0.4 * steer + 0.6 * self.prev_steer
        self.prev_steer = steer
        self.last_steer = steer
        return steer

    def get_pwm_pair(self, steer):

        PWM_center = 1500
        PWM_range  = 200

        s = np.clip(steer, -1, 1)
        mag = abs(s)

        t = mag ** 0.5
        m = t * t * (3 - 2 * t)

        delta = m * PWM_range

        if s >= 0:
            pwm_L = PWM_center - delta
            pwm_R = PWM_center + delta
        else:
            pwm_L = PWM_center + delta
            pwm_R = PWM_center - delta

        pwm_L = int(np.clip(pwm_L, 1300, 1700))
        pwm_R = int(np.clip(pwm_R, 1300, 1700))

        return pwm_L, pwm_R

    def collide(self):
        bx, by = self.boat_pos

        for ox, oy, r in self.obstacles:
            if abs(ox - bx) > 150:
                continue
            if abs(oy - by) > 150:
                continue
            if (bx-ox)**2 + (by-oy)**2 <= (r+self.boat_radius)**2:
                return True
        return False

    def draw_text(self, text, x, y):
        outline = self.font.render(text, True, (0,0,0))
        self.screen.blit(outline, (x-1, y-1))
        self.screen.blit(outline, (x+1, y-1))
        self.screen.blit(outline, (x-1, y+1))
        self.screen.blit(outline, (x+1, y+1))

        white = self.font.render(text, True, (255,255,255))
        self.screen.blit(white, (x, y))

    def render(self, desired_heading):

        self.screen.fill((235,235,235))
        self.screen.blit(self.trail_surface, (0,0))

        for ox, oy, r in self.obstacles:
            pygame.draw.circle(self.screen, (80,80,80), (int(ox), int(oy)), r)

        pygame.draw.circle(
            self.screen, (255,0,0),
            (int(self.target[0]), int(self.target[1])), 26
        )

        bx = int(self.boat_pos[0])
        by = int(self.boat_pos[1])
        h = self.boat_heading

        p1 = (bx + math.cos(h)*22, by + math.sin(h)*22)
        p2 = (bx + math.cos(h+2.5)*16, by + math.sin(h+2.5)*16)
        p3 = (bx + math.cos(h-2.5)*16, by + math.sin(h-2.5)*16)

        pygame.draw.polygon(
            self.screen, (0,90,255),
            [(int(p1[0]),int(p1[1])),
             (int(p2[0]),int(p2[1])),
             (int(p3[0]),int(p3[1]))]
        )

        deg_des_old = math.degrees(self.last_desired)
        deg_cur_old = math.degrees(self.boat_heading)

        deg_des = (deg_des_old + 90) % 360
        deg_cur = (deg_cur_old + 90) % 360

        hud_x = 20
        hud_y = 20
        gap = 25

        self.draw_text(f"Desired Heading : {deg_des-180: .0f} deg", hud_x, hud_y); hud_y += gap
        self.draw_text(f"Current Heading : {deg_cur: .0f} deg", hud_x, hud_y); hud_y += gap
        self.draw_text(f"PWM Left       : {self.last_pwm_L}", hud_x, hud_y); hud_y += gap
        self.draw_text(f"PWM Right      : {self.last_pwm_R}", hud_x, hud_y); hud_y += gap

        pygame.display.update()
        self.clock.tick(60)

    def save_screenshot(self):
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{now}.png"
        pygame.image.save(self.screen, filename)

    def increment_frame(self):
        self.frame_count += 1

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

        desired = env.compute_safe_heading(gps_heading, lidar)

        env.heading_hist.append(desired)
        if len(env.heading_hist) > env.heading_hist_len:
            env.heading_hist.pop(0)

        desired = math.atan2(
            np.mean([math.sin(a) for a in env.heading_hist]),
            np.mean([math.cos(a) for a in env.heading_hist])
        )

        steer = env.heading_control(desired, env.boat_heading, env.boat_ang_vel)
        pwm_L, pwm_R = env.get_pwm_pair(steer)

        env.step(pwm_L, pwm_R)
        env.render(desired)

        if env.collide() or np.linalg.norm(env.target - env.boat_pos) < 70:
            env.save_screenshot()
            env.reset()

if __name__ == "__main__":
    run()
