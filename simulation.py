import pygame
import numpy as np
import math
import random
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import datetime
import os

# --- Constants ---
GRID = 8
GRID_W = 2000 // GRID
GRID_H = 600 // GRID

# --- Math & Path Planning Functions ---

def bezier_point(p0, p1, p2, t):
    return (1-t)*(1-t)*p0 + 2*(1-t)*t*p1 + t*t*p2

def build_bezier_path(p0, p1, p2, samples=40):
    ts = np.linspace(0, 1, samples)
    pts = [bezier_point(p0, p1, p2, t) for t in ts]
    return np.array(pts, dtype=np.float32)

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def cubic_bezier(p0, p1, p2, p3, n=90):
    t = np.linspace(0, 1, n)
    T = t[:, None]
    B = (1-T)**3 * p0 + 3*(1-T)**2*T*p1 + 3*(1-T)*T**2*p2 + T**3*p3
    return B

def make_bezier_path(boat_pos, boat_heading, goal):
    d = np.linalg.norm(goal - boat_pos)
    if d < 1:
        return None

    p0 = boat_pos.copy()
    forward = np.array([math.cos(boat_heading), math.sin(boat_heading)])
    p1 = boat_pos + forward * min(120, d * 0.4)

    p3 = goal.copy()
    v_goal = p3 - p1
    # Avoid division by zero
    norm_v_goal = np.linalg.norm(v_goal)
    if norm_v_goal < 1e-6:
        v_goal_n = np.zeros(2)
    else:
        v_goal_n = v_goal / norm_v_goal
        
    p2 = p3 - v_goal_n * min(120, d * 0.4)

    return cubic_bezier(p0, p1, p2, p3, n=90)

def pure_pursuit(path, boat_pos, lookahead=70):
    if path is None:
        return None

    # lookahead보다 멀리 있는 첫 번째 점을 찾습니다.
    for i in range(len(path)-1):
        p = path[i]
        if np.linalg.norm(p - boat_pos) > lookahead:
            return p
    
    return path[-1]

def find_pp_target(path, pos, L=80):
    for i in range(len(path)-1):
        if np.linalg.norm(path[i]-pos) >= L:
            return path[i]
    return path[-1]

# --- Lidar & Grid Functions ---

def lidar_hits_np(boat_pos, boat_heading, rel_angles, obstacles, lidar_range):
    if len(obstacles) == 0:
        n = len(rel_angles)
        d = np.full(n, lidar_range, np.float32)
        return d, [None]*n
    ox = obstacles[:, 0]
    oy = obstacles[:, 1]
    orad = obstacles[:, 2]
    angs = boat_heading + rel_angles
    dx = np.cos(angs)
    dy = np.sin(angs)
    d_final = np.full_like(angs, lidar_range, dtype=np.float32)
    hits = [None] * len(angs)
    
    x0 = boat_pos[0]
    y0 = boat_pos[1]

    for i in range(len(angs)):
        vx = dx[i]
        vy = dy[i]
        px = ox - x0
        py = oy - y0
        b = px*vx + py*vy
        mask = b > 0
        if not np.any(mask):
            continue
        
        b2 = b[mask]
        px2 = px[mask]
        py2 = py[mask]
        r2 = orad[mask]
        
        perp = px2 - b2*vx
        perp2 = py2 - b2*vy
        dist2 = perp*perp + perp2*perp2
        hitmask = dist2 <= r2*r2
        
        if not np.any(hitmask):
            continue
            
        b3 = b2[hitmask]
        dist2_2 = dist2[hitmask]
        r3 = r2[hitmask]
        
        # Sqrt domain check done implicitly by selection logic logic usually, 
        # but r3*r3 - dist2_2 should be >= 0 due to hitmask
        f = np.sqrt(np.maximum(0, r3*r3 - dist2_2))
        t = b3 - f
        
        # Filter valid hits within current min distance
        valid_t = t[(t > 0) & (t < d_final[i])]
        
        if len(valid_t) > 0:
            tmin = np.min(valid_t)
            d_final[i] = tmin
            hx = x0 + vx * tmin
            hy = y0 + vy * tmin
            hits[i] = (hx, hy)
            
    return d_final, hits

def init_grid():
    return np.zeros((GRID_H, GRID_W), dtype=np.float32)

def update_grid(grid, hits):
    for p in hits:
        if p is None: continue
        gx = int(p[0] // GRID)
        gy = int(p[1] // GRID)
        if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
            grid[gy, gx] = min(grid[gy, gx] + 1.0, 20.0)

def extract_clusters_from_grid(grid):
    OCC = 2.0
    ys, xs = np.where(grid >= OCC)
    if len(xs) == 0:
        return []
    pts = np.column_stack([(xs * GRID + GRID/2), (ys * GRID + GRID/2)]).astype(np.float32)
    if len(pts) == 0:
        return []
    
    # DBSCAN clustering
    db = DBSCAN(eps=22, min_samples=2).fit(pts)
    labels = db.labels_
    clusters = []
    for lb in set(labels):
        if lb == -1:
            continue
        mask = (labels == lb)
        clusters.append(np.mean(pts[mask], axis=0))
    return clusters

def match_clusters(prev_clusters, prev_ids, new_clusters):
    if len(prev_clusters) == 0:
        return new_clusters, list(range(len(new_clusters)))
    
    cell_prev = {}
    for cid, c in zip(prev_ids, prev_clusters):
        key = (int(c[0] // GRID), int(c[1] // GRID))
        cell_prev[key] = cid
        
    new_ids = []
    maxid = max(prev_ids) + 1 if prev_ids else 0
    
    for c in new_clusters:
        key = (int(c[0] // GRID), int(c[1] // GRID))
        if key in cell_prev:
            new_ids.append(cell_prev[key])
        else:
            new_ids.append(maxid)
            maxid += 1
            
    return new_clusters, new_ids

# --- High Level Logic / Gap Finding ---

def front_is_clear(boat_pos, boat_heading, obstacles, check_dist=300, fov=np.deg2rad(25)):
    bx, by = boat_pos
    for (ox, oy, r) in obstacles:
        dx = ox - bx
        dy = oy - by
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > check_dist:
            continue
        ang = math.atan2(dy, dx)
        rel = wrap(ang - boat_heading)
        if abs(rel) < fov:
            return False
    return True

def find_gap(clusters, ids, boat_pos, boat_heading, gps_heading, visited, grid, obstacles):
    # 전방이 깨끗하면 굳이 갭을 찾지 않음 (직진 선호)
    if front_is_clear(boat_pos, boat_heading, obstacles):
        return None
        
    bx, by = boat_pos
    gps_vec = np.array([math.cos(gps_heading), math.sin(gps_heading)])
    
    # 1. 시야 내의 클러스터(장애물) 수집
    items = []
    for i, c in enumerate(clusters):
        v = c - boat_pos
        dist = np.linalg.norm(v)
        ang = wrap(math.atan2(v[1], v[0]) - boat_heading)
        # 전방 180도 내에 있는 것만 고려
        if abs(ang) < np.pi/2:
            items.append((ang, dist, c, ids[i]))
            
    if len(items) < 2:
        return None
        
    # 각도 순 정렬
    items.sort(key=lambda x: x[0])
    
    # 2. 갭 찾기
    gaps = []
    for i in range(len(items) - 1):
        # 장애물 사이 각도 차이가 2도 이상이면 갭으로 간주
        if (items[i+1][0] - items[i][0]) > np.deg2rad(2.0):
            gaps.append((i, i+1))
            
    if not gaps:
        return None
        
    best = None
    best_sc = -1
    ox = obstacles[:, 0]
    oy = obstacles[:, 1]
    
    for gi, gj in gaps:
        ang1, d1, c1, id1 = items[gi]
        ang2, d2, c2, id2 = items[gj]
        
        # 이미 방문했던(지나온) 갭은 무시
        if (id1, id2) in visited or (id2, id1) in visited:
            continue
            
        mid = (c1 + c2) / 2
        mx, my = mid
        rel = mid - boat_pos
        distm = np.linalg.norm(rel) + 1e-6
        
        # 그리드 상에서 막혀있는지 확인 (간단한 체크)
        gx = int(mx // GRID)
        gy = int(my // GRID)
        blocked = False
        for dy_grid in range(-2, 3):
            for dx_grid in range(-2, 3):
                yy = gy + dy_grid
                xx = gx + dx_grid
                if 0 <= xx < GRID_W and 0 <= yy < GRID_H:
                    if grid[yy, xx] >= 3.0:
                        blocked = True
                        break
            if blocked: break
        if blocked: continue
        
        # 점수 계산 (Heuristic)
        ang_mid = math.atan2(rel[1], rel[0])
        ang_err = wrap(ang_mid - gps_heading)
        
        heading_align = math.exp(-(ang_err / 0.9)**2)
        
        forward_proj = np.dot(rel / distm, gps_vec)
        forward_proj = max(forward_proj, 0)**1.5
        
        lateral = abs(ang2 - ang1) / (np.pi/2)
        lateral = min(max(lateral, 0), 1)**2
        
        sym = 1 - abs(abs(ang1) - abs(ang2)) / (np.pi/2)
        sym = min(max(sym, 0), 1)
        
        lateral_full = 0.6 * lateral + 0.4 * sym
        
        # 실제 장애물과의 충돌 가능성 체크
        vx = mx - bx
        vy = my - by
        seg2 = distm * distm
        d2_obs = (ox - bx)**2 + (oy - by)**2
        
        # 근처 장애물만 필터링
        mask = d2_obs <= (distm + 200)**2
        obs_f = obstacles[mask]
        
        min_clear = 9999
        for (ox2, oy2, r2) in obs_f:
            px = ox2 - bx
            py = oy2 - by
            # 선분과 점 사이 거리
            t = (px*vx + py*vy) / seg2
            t = max(0, min(1, t))
            cx = bx + t*vx
            cy = by + t*vy
            d = math.sqrt((ox2 - cx)**2 + (oy2 - cy)**2) - r2
            if d < min_clear:
                min_clear = d
                
        min_clear = max(min_clear, 0)
        path_clear = min(min_clear / 120, 1)**2.5
        
        # 클러스터 밀집도 패널티
        cnt = 0
        for (ox2, oy2, r2) in obs_f:
            if (ox2 - mx)**2 + (oy2 - my)**2 < 100*100:
                cnt += 1
        cluster_pen = math.exp(-0.5 * cnt)
        
        gap_w = np.linalg.norm(c2 - c1)
        width_w = min(gap_w / 90, 1)
        small_gap = math.exp(-gap_w / 40)
        
        sc = heading_align**4.5 * forward_proj**1.5 * lateral_full**2 * path_clear**2.5 * width_w**1.2 * cluster_pen * small_gap
        
        if sc > best_sc:
            best_sc = sc
            best = {"pos": mid, "c1": c1.copy(), "c2": c2.copy(), "pair": (id1, id2), "score": sc}
            
    return best

def reactive_avoidance(dists, angles):
    SAFE = 360
    sigma = 120
    a = 0.
    for d, ang in zip(dists, angles):
        if d < SAFE:
            w = math.exp(-(d / sigma)**2)
            # 정면일수록 회피 강도 증가
            front = max(1.1 - abs(ang) / (math.pi/2), 0.4)
            a -= w * front * math.sin(ang)
    return a

# --- Main Class ---

class BoatEnv:
    def __init__(self):
        pygame.init()
        self.w = 2000
        self.h = 600
        self.screen = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()
        self.dt = 0.04
         
        self.lidar_beams = 90
        self.lidar_range = 350
        self.rel_angles = np.linspace(-np.pi/2, np.pi/2, self.lidar_beams)
        
        self.mass = 20
        self.inertia = 0.10
        self.drag = 0.38
        self.rot_drag = 0.55
        self.boat_radius = 25
        
        self.trail = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        self.path_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        
        self.obs_n = 80
        self.obs_r = 17
        self.min_obs = 125
        
        self.grid = init_grid()
        self.clusters = []
        self.cluster_ids = []
        self.current_wp = None
        self.visited = set()
        
        self.frame = 0
        self.prev_steer = 0
        self.wp_check_timer = 0
        self.steer_timer = 0
        self.path_timer = 0
        
        self.bezier_path = None
        self.pursuit_target = None
        
        self.reset()

    def reset(self):
        self.boat_pos = np.array([65, self.h/2], dtype=np.float32)
        self.boat_vel = np.zeros(2)
        self.boat_ang_vel = 0
        self.target = np.array([self.w - 200, self.h/2], dtype=np.float32)
        
        self.trail.fill((0, 0, 0, 0))
        self.path_surf.fill((0, 0, 0, 0))
        
        obs = []
        t = 0
        while len(obs) < self.obs_n and t < 5000:
            t += 1
            x = random.randint(300, self.w - 300)
            y = random.randint(30, self.h - 30)
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
        
        dx = self.target[0] - self.boat_pos[0]
        dy = self.target[1] - self.boat_pos[1]
        self.boat_heading = math.atan2(dy, dx)
        
        self.grid[:] = 0
        self.clusters = []
        self.cluster_ids = []
        self.current_wp = None
        self.visited = set()
        
        self.wp_check_timer = 0
        self.steer_timer = 0
        self.path_timer = 0
        self.bezier_path = None
        self.pursuit_target = None

    def pwm_to_thrust(self, p):
        return p * 10

    def step(self, L, R):
        tL = self.pwm_to_thrust(L)
        tR = self.pwm_to_thrust(R)
        fwd = tL + tR
        mom = (tR - tL) * 0.006
        
        hv = np.array([math.cos(self.boat_heading), math.sin(self.boat_heading)])
        
        acc = fwd / self.mass
        # Drag calculation
        vel_norm = np.linalg.norm(self.boat_vel)
        if vel_norm > 0:
            drag = -self.drag * vel_norm * self.boat_vel
        else:
            drag = np.zeros(2)
            
        prev = self.boat_pos.copy()
        self.boat_vel += (acc * hv + drag) * self.dt
        self.boat_pos += self.boat_vel * self.dt
        
        if self.frame % 7 == 0:
            pygame.draw.line(self.trail, (0, 120, 255, 255),
                             (int(prev[0]), int(prev[1])),
                             (int(self.boat_pos[0]), int(self.boat_pos[1])), 2)
                             
        ang_acc = (mom - self.rot_drag * self.boat_ang_vel) / self.inertia
        self.boat_ang_vel += ang_acc * self.dt
        self.boat_ang_vel *= 0.84
        self.boat_heading += self.boat_ang_vel * self.dt

    def collide(self):
        ox = self.obstacles[:, 0]
        oy = self.obstacles[:, 1]
        rr = self.obstacles[:, 2] + self.boat_radius
        dx = ox - self.boat_pos[0]
        dy = oy - self.boat_pos[1]
        hit = np.any(dx*dx + dy*dy <= rr*rr)
        wall = (self.boat_pos[0] <= 0 or self.boat_pos[0] >= self.w or
                self.boat_pos[1] <= 0 or self.boat_pos[1] >= self.h)
        return hit or wall

    def get_pwm(self, steer):
        dead = 0.03
        if abs(steer) < dead:
            steer = 0
        mid = 1500
        rng = 210
        m = np.log1p(4 * abs(steer)) / np.log(5)
        d = m * rng
        if steer >= 0:
            L = mid - d; R = mid + d
        else:
            L = mid + d; R = mid - d
        return int(np.clip(L, 1300, 1700)), int(np.clip(R, 1300, 1700))

    def validate_wp_grid(self):
        if self.current_wp is None:
            return
        self.wp_check_timer += self.dt
        if self.wp_check_timer < 0.20:
            return
        self.wp_check_timer = 0
        
        wp = self.current_wp["pos"]
        pair = self.current_wp["pair"]
        gx = int(wp[0] // GRID)
        gy = int(wp[1] // GRID)
        rad = int(35 // GRID)
        
        for yy in range(max(0, gy - rad), min(GRID_H, gy + rad + 1)):
            for xx in range(max(0, gx - rad), min(GRID_W, gx + rad + 1)):
                if self.grid[yy, xx] >= 3:
                    self.visited.add(pair)
                    self.visited.add((pair[1], pair[0]))
                    self.current_wp = None
                    return

    def validate_wp_obstacle_5x5(self):
        if self.current_wp is None:
            return
        wp = self.current_wp["pos"]
        gx = int(wp[0] // GRID)
        gy = int(wp[1] // GRID)
        xs = range(gx - 2, gx + 3)
        ys = range(gy - 2, gy + 3)
        
        ox = self.obstacles[:, 0]
        oy = self.obstacles[:, 1]
        rr = self.obstacles[:, 2]
        
        for yy in ys:
            for xx in xs:
                if 0 <= xx < GRID_W and 0 <= yy < GRID_H:
                    cx = xx * GRID + GRID * 0.5
                    cy = yy * GRID + GRID * 0.5
                    dx = ox - cx
                    dy = oy - cy
                    # 그리드 셀 중심과 장애물들 거리 체크
                    hit = np.any(dx*dx + dy*dy <= rr*rr)
                    if hit:
                        p = self.current_wp["pair"]
                        self.visited.add(p)
                        self.visited.add((p[1], p[0]))
                        self.current_wp = None
                        return

    def update_steering(self, dists):
        self.steer_timer += self.dt
        if self.steer_timer < 0.10:
            return None
        self.steer_timer = 0
        
        if self.pursuit_target is None:
            return 0
            
        px, py = self.pursuit_target
        heading_target = math.atan2(py - self.boat_pos[1], px - self.boat_pos[0])
        heading_error = wrap(heading_target - self.boat_heading)
        
        # [유지] 웨이포인트 추적 강도
        k = heading_error * 0.6

        steer_raw = k
        
        # [유지] 부드러운 조향
        steer_f = 0.3 * steer_raw + 0.7 * self.prev_steer
        self.prev_steer = steer_f
        
        avoid = reactive_avoidance(dists, self.rel_angles)

        steer = steer_f + 0.008 * avoid 
        
        steer = np.clip(steer, -1, 1)
        return steer

    def render(self, hits):
        self.screen.fill((180, 220, 255))
        self.screen.blit(self.trail, (0, 0))
        self.screen.blit(self.path_surf, (0, 0))
        
        # 장애물 그리기
        for ox, oy, r in self.obstacles:
            pygame.draw.circle(self.screen, (50, 50, 50), (int(ox), int(oy)), int(r + 2))
            pygame.draw.circle(self.screen, (255, 255, 255), (int(ox), int(oy)), int(r))
            
        # 그리드 점유 그리기
        occ = np.where(self.grid >= 3)
        for gy, gx in zip(occ[0], occ[1]):
            x = gx * GRID
            y = gy * GRID
            pygame.draw.rect(self.screen, (150, 150, 150), (x, y, GRID, GRID))
            
        # Lidar hits
        for p in hits:
            if p is not None:
                pygame.draw.circle(self.screen, (255, 0, 0), (int(p[0]), int(p[1])), 2)
                
        # Target
        pygame.draw.circle(self.screen, (200, 0, 0), (int(self.target[0]), int(self.target[1])), 8)
        
        # Waypoint
        if self.current_wp is not None:
            wp = self.current_wp
            pygame.draw.circle(self.screen, (0, 0, 200), (int(wp["pos"][0]), int(wp["pos"][1])), 7)
            pygame.draw.line(self.screen, (0, 200, 0),
                             (int(wp["c1"][0]), int(wp["c1"][1])),
                             (int(wp["c2"][0]), int(wp["c2"][1])), 4)
                             
        # Bezier Path
        if self.bezier_path is not None:
            for i in range(len(self.bezier_path) - 1):
                x1, y1 = self.bezier_path[i]
                x2, y2 = self.bezier_path[i+1]
                pygame.draw.line(self.path_surf, (0, 180, 0), (int(x1), int(y1)), (int(x2), int(y2)), 2)
                
        # Pursuit Target
        if self.pursuit_target is not None:
            px, py = self.pursuit_target
            pygame.draw.circle(self.screen, (0, 0, 255), (int(px), int(py)), 5)
            
        # Boat
        bx, by = self.boat_pos
        h = self.boat_heading
        ch, sh = math.cos(h), math.sin(h)
        GAP = 10; L = 80; W = 15
        left = (bx - sh*GAP, by + ch*GAP)
        right = (bx + sh*GAP, by - ch*GAP)
        
        hull = [(L*0.50, 0), (L*0.10, W),
                (-L*0.30, W*0.8), (-L*0.48, W*0.55),
                (-L*0.50, 0), (-L*0.48, -W*0.55),
                (-L*0.30, -W*0.8), (L*0.10, -W)]
                
        def TR(c, px, py):
            return int(c[0] + px*ch - py*sh), int(c[1] + px*sh + py*ch)
            
        left_h = [TR(left, p[0], p[1]) for p in hull]
        right_h = [TR(right, p[0], p[1]) for p in hull]
        
        pygame.draw.polygon(self.screen, (0, 0, 240), left_h)
        pygame.draw.polygon(self.screen, (0, 0, 240), right_h)
        
        pygame.display.update()
        self.clock.tick(60)

def run():
    env = BoatEnv()

    while True:
        env.frame += 1

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                return

        dists, hits = lidar_hits_np(
            env.boat_pos, env.boat_heading,
            env.rel_angles, env.obstacles,
            env.lidar_range
        )

        update_grid(env.grid, hits)
        env.grid *= 0.945

        new_c = extract_clusters_from_grid(env.grid)
        env.clusters, env.cluster_ids = match_clusters(
            env.clusters, env.cluster_ids, new_c
        )

        dx = env.target[0] - env.boat_pos[0]
        dy = env.target[1] - env.boat_pos[1]
        gps_head = math.atan2(dy, dx)

        new_wp = find_gap(
            env.clusters, env.cluster_ids,
            env.boat_pos, env.boat_heading,
            gps_head, env.visited,
            env.grid, env.obstacles
        )

        if env.current_wp is not None:
            dnow = np.linalg.norm(env.current_wp["pos"] - env.boat_pos)
            if dnow < 22:
                p = env.current_wp["pair"]
                env.visited.add(p)
                env.visited.add((p[1], p[0]))
                env.current_wp = None

        if new_wp is not None:
            if env.current_wp is None:
                env.current_wp = new_wp
            else:
                # --- 안전장치 시작 ---
                
                # 1. 거리 체크: 현재 목표에 이미 특정 px 이내로 근접했다면 바꾸지 않음 (그냥 가서 찍는게 안전함)
                dist_to_curr = np.linalg.norm(env.current_wp["pos"] - env.boat_pos)
                
                if dist_to_curr > 100:
                    
                    # 2. 각도 차이 계산: 현재 가려던 곳 vs 새로 발견한 곳의 각도 차이
                    vec_curr = env.current_wp["pos"] - env.boat_pos
                    vec_new = new_wp["pos"] - env.boat_pos
                    
                    ang_curr = math.atan2(vec_curr[1], vec_curr[0])
                    ang_new = math.atan2(vec_new[1], vec_new[0])
                    
                    angle_diff = abs(wrap(ang_new - ang_curr))

                    threshold = 1
                    
                    # 만약 방향을 45도(0.78rad) 이상 확 꺾어야 한다면, 
                    # 점수가 1.5배(50%) 이상 좋을 때만 바꿈 (위험하니까)
                    if angle_diff > np.deg2rad(45):
                        threshold = 1.2
                    
                    # 만약 방향을 90도 가까이 꺾어야 한다면 거의 안 바꿈
                    if angle_diff > np.deg2rad(80):
                        threshold = 2

                    # 최종 비교
                    if new_wp["score"] > env.current_wp["score"] * threshold:
                        env.current_wp = new_wp
        
        dist_to_final = np.linalg.norm(env.target - env.boat_pos)
        
        # [추가] X축 거리 계산
        dist_x = abs(env.target[0] - env.boat_pos[0])

        # 1. 수정된 조건: 실제 거리가 멀어도, X축으로 200px 이내로 좁혀지면 진입
        if dist_x < 200:
            
            # 2. 목표지점의 상대 각도 계산
            dx_t = env.target[0] - env.boat_pos[0]
            dy_t = env.target[1] - env.boat_pos[1]
            t_angle = math.atan2(dy_t, dx_t)
            rel_t = wrap(t_angle - env.boat_heading)
            
            # 3. 목표가 전방 시야(Lidar 범위) 안에 있는지 확인
            if abs(rel_t) < np.pi/2:
                # 4. 해당 각도의 Lidar 인덱스 찾기
                beam_idx = int((rel_t + np.pi/2) / np.pi * (env.lidar_beams - 1))
                beam_idx = np.clip(beam_idx, 0, env.lidar_beams - 1)
                
                # 5. 해당 방향에 장애물 확인
                is_clear = True
                check_range = 2
                for i in range(beam_idx - check_range, beam_idx + check_range + 1):
                    if 0 <= i < env.lidar_beams:
                        # 주의: 여기서 비교할 때는 여전히 '실제 직선 거리(dist_to_final)'를 써야 정확합니다.
                        if dists[i] < dist_to_final: 
                            is_clear = False
                            break
                
                # 6. 장애물이 없으면 웨이포인트 취소
                if is_clear:
                    env.current_wp = None

        if env.current_wp is None:
            goal = env.target
        else:
            goal = env.current_wp["pos"]

        env.path_timer += env.dt
        if env.path_timer >= 0.05:
            env.path_timer = 0
            env.path_surf.fill((0, 0, 0, 0))
            env.bezier_path = make_bezier_path(env.boat_pos, env.boat_heading, goal)
            if env.bezier_path is not None:
                env.pursuit_target = pure_pursuit(env.bezier_path, env.boat_pos, lookahead=70)

        steer = env.update_steering(dists)
        if steer is None:
            steer = 0

        L, R = env.get_pwm(steer)
        env.step(L, R)

        env.validate_wp_grid()
        env.validate_wp_obstacle_5x5()

        env.render(hits)

        # 충돌 혹은 목표 도달 시 스크린샷 및 리셋
        if env.collide() or np.linalg.norm(env.target - env.boat_pos) < 70:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # 경로 수정 필요 (Windows 기준 예시)
            outdir = r"C:\Reinforce Learning\screenshot"
            if not os.path.exists(outdir):
                try:
                    os.makedirs(outdir)
                except:
                    pass
            p = os.path.join(outdir, f"{ts}.png")
            try:
                pygame.image.save(env.screen, p)
            except:
                print("Screenshot save failed")
            env.reset()

if __name__ == "__main__":
    run()