import numpy as np
import math
import random
import csv
import multiprocessing
from multiprocessing import Pool


class BoatEnv:
    def __init__(self,
                 avoid_scale=2.2,
                 gps_gain=10.0,
                 gap_gain=9.0,
                 map_w=2000,
                 map_h=600,
                 lidar_beams=90,
                 lidar_range=450,
                 dt=0.04):

        self.avoid_scale = avoid_scale
        self.gps_gain = gps_gain
        self.gap_gain = gap_gain

        self.map_w = map_w
        self.map_h = map_h
        self.dt = dt

        self.lidar_beams = lidar_beams
        self.lidar_range = lidar_range

        self.rel_angles = np.linspace(-np.pi/2, np.pi/2, self.lidar_beams)
        self.rel_cos = np.cos(self.rel_angles)
        self.rel_sin = np.sin(self.rel_angles)

        self.mass = 20.0
        self.inertia = 0.08
        self.drag = 0.40
        self.rot_drag = 0.60
        self.boat_radius = 25
        self.obs_radius = 10

        self.obs_count = 100
        self.min_obs_dist = 110

        self.prev_steer = 0.0
        self.prev_gap_dir = 0

        self.reset()

    def reset(self):

        self.boat_pos = np.array([70.0, self.map_h/2], dtype=np.float32)
        self.boat_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.boat_ang_vel = 0.0

        self.target = np.array([self.map_w - 200.0, self.map_h/2], dtype=np.float32)

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

        self.obstacles_np = np.array([[o[0], o[1], o[2]] for o in self.obstacles], dtype=np.float32)

    def pwm_to_thrust(self, pwm):
        return pwm * 3.0

    def step(self, pwm_L, pwm_R):

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

        ang_acc = (moment - self.rot_drag * self.boat_ang_vel) / self.inertia
        self.boat_ang_vel += ang_acc * self.dt
        self.boat_ang_vel = np.clip(self.boat_ang_vel, -1.2, 1.2)
        self.boat_heading += self.boat_ang_vel * self.dt

    def lidar_scan(self):

        bx, by = self.boat_pos
        ox = self.obstacles_np[:, 0]
        oy = self.obstacles_np[:, 1]
        rad = self.obstacles_np[:, 2]

        dx = ox - bx
        dy = oy - by

        dist2 = dx*dx + dy*dy
        dist = np.sqrt(dist2)
        ang_obs = np.arctan2(dy, dx)

        obs_radius = rad + self.boat_radius

        beam_angles = self.boat_heading + self.rel_angles
        lidar_out = np.full(self.lidar_beams, self.lidar_range, dtype=np.float32)

        for i in range(self.lidar_beams):
            ang = beam_angles[i]
            da = ((ang_obs - ang + np.pi) % (2*np.pi)) - np.pi
            close = np.abs(da) < 0.15

            if np.any(close):
                idx = np.where(close)[0]
                cx = ox[idx]
                cy = oy[idx]
                dx_i = cx - bx
                dy_i = cy - by
                dirx = math.cos(ang)
                diry = math.sin(ang)

                proj = dx_i*dirx + dy_i*diry
                hit_mask = proj > 0
                if np.any(hit_mask):
                    proj2 = proj[hit_mask]
                    obx = dx_i[hit_mask]
                    oby = dy_i[hit_mask]
                    perp = obx*diry - oby*dirx
                    perp = np.abs(perp)
                    rad2 = obs_radius[idx][hit_mask]
                    hit_final = perp <= rad2
                    if np.any(hit_final):
                        proj3 = proj2[hit_final]
                        dmin = np.min(proj3)
                        if dmin < lidar_out[i]:
                            lidar_out[i] = dmin

        return lidar_out

    def avoidance_strength(self, d):
        x = (self.lidar_range*1.2 - d) / (self.lidar_range*1.2)
        return x * self.avoid_scale

    def compute_safe_heading(self, gps_heading, lidar):

        cosH = math.cos(self.boat_heading)
        sinH = math.sin(self.boat_heading)

        A = np.array([math.cos(gps_heading),
                      math.sin(gps_heading)]) * self.gps_gain

        rep = (self.lidar_range*1.2 - lidar) / (self.lidar_range*1.2)
        rep = np.clip(rep, 0, 1) * self.avoid_scale

        rx = cosH*self.rel_cos - sinH*self.rel_sin
        ry = sinH*self.rel_cos + cosH*self.rel_sin

        R = np.array([
            -np.sum(rep * rx),
            -np.sum(rep * ry)
        ])

        left_gap = np.mean(lidar[:self.lidar_beams//2])
        right_gap = np.mean(lidar[self.lidar_beams//2:])
        diff = right_gap - left_gap

        L_side = np.mean(lidar[:10])
        R_side = np.mean(lidar[-10:])
        diff2 = R_side - L_side

        combined = 0.7*(diff/self.lidar_range) + 0.3*(diff2/self.lidar_range)

        if abs(combined) < 0.30:
            gap_dir = self.prev_gap_dir
        else:
            gap_dir = 1 if combined > 0 else -1

        self.prev_gap_dir = gap_dir

        G = np.array([
            math.cos(self.boat_heading + 0.8*gap_dir),
            math.sin(self.boat_heading + 0.8*gap_dir)
        ]) * self.gap_gain

        V = A + R + G
        return math.atan2(V[1], V[0])

    def heading_control(self, desired, current, ang_vel):

        err = (desired - current + np.pi) % (2*np.pi) - np.pi
        steer = 1.4*err - 0.25*ang_vel
        steer = np.clip(steer, -1, 1)
        steer = 0.4*steer + 0.6*self.prev_steer
        self.prev_steer = steer
        return steer

    def get_pwm_pair(self, steer):

        PWM_center = 1500
        PWM_range = 200
        s = np.clip(steer, -1, 1)
        mag = abs(s)

        t = mag**0.5
        m = t*t*(3 - 2*t)
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
        ox = self.obstacles_np[:, 0]
        oy = self.obstacles_np[:, 1]
        r = self.obstacles_np[:, 2] + self.boat_radius

        dx = ox - bx
        dy = oy - by
        d2 = dx*dx + dy*dy
        return np.any(d2 <= r*r)


def simulate_once(a, g, k):

    env = BoatEnv(a, g, k)

    while True:

        lidar = env.lidar_scan()

        dx = env.target[0] - env.boat_pos[0]
        dy = env.target[1] - env.boat_pos[1]
        gps_heading = math.atan2(dy, dx)

        desired = env.compute_safe_heading(gps_heading, lidar)
        steer = env.heading_control(desired, env.boat_heading, env.boat_ang_vel)
        pwm_L, pwm_R = env.get_pwm_pair(steer)

        env.step(pwm_L, pwm_R)

        if env.collide():
            return 0

        if np.linalg.norm(env.target - env.boat_pos) < 70:
            return 1


def batch_simulation_parallel():

    avoid_values = [1.0 + 0.2*i for i in range(10)]
    gps_values = [5.0 + i for i in range(10)]
    gap_values = [3.0 + i for i in range(5)]

    params = []
    for a in avoid_values:
        for g in gps_values:
            for k in gap_values:
                for _ in range(10):
                    params.append((a, g, k))

    num_cores = multiprocessing.cpu_count()
    print("Detected CPU cores:", num_cores)

    with Pool(processes=num_cores) as p:
        results = p.starmap(simulate_once, params)

    idx = 0
    f = open("results.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(["avoid_scale", "gps_gain", "gap_gain", "success_rate"])

    for a in avoid_values:
        for g in gps_values:
            for k in gap_values:
                trials = results[idx:idx+10]
                idx += 10
                suc = sum(trials) / 10.0
                print(a, g, k, suc)
                w.writerow([a, g, k, suc])

    f.close()
    print("Saved results.csv")


if __name__ == "__main__":
    batch_simulation_parallel()
