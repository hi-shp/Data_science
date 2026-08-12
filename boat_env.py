import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import random
import csv


class BoatEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=False):
        super(BoatEnv, self).__init__()
        self.render_mode = render_mode

        # 경기장 크기
        self.WIDTH, self.HEIGHT = 1000, 200

        # 행동: 좌회전(0), 직진(1), 우회전(2)
        self.action_space = spaces.Discrete(3)

        # 상태: LiDAR 9개 + 목표 거리 + 방향오차 = 총 11차원
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

        # 렌더링 설정
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Boat Navigation with LiDAR")
            self.clock = pygame.time.Clock()

        # 데이터 저장용 리스트
        self.log_data = []

        self.reset()

    # LiDAR 스캔 함수
    def _get_lidar_scan(self, num_rays=9, max_range=150):
        scans = []
        angle_span = np.pi / 2  # 전방 ±90도
        start_angle = self.theta - angle_span / 2
        for i in range(num_rays):
            ray_angle = start_angle + i * (angle_span / (num_rays - 1))
            distance = max_range
            for r in np.linspace(0, max_range, 100):
                rx = self.x + np.cos(ray_angle) * r
                ry = self.y + np.sin(ray_angle) * r
                # 벽 충돌
                if ry <= 0 or ry >= self.HEIGHT:
                    distance = r
                    break
                # 장애물 충돌
                for (ox, oy) in self.obstacles:
                    if np.hypot(rx - ox, ry - oy) < 10:
                        distance = r
                        break
                if distance < max_range:
                    break
            scans.append(distance / max_range)  # 0~1 정규화
        return np.array(scans, dtype=np.float32)

    # 상태(state) 구성
    def _get_state(self):
        lidar = self._get_lidar_scan()
        dx, dy = self.goal - np.array([self.x, self.y])
        distance = np.hypot(dx, dy)
        heading_error = math.atan2(dy, dx) - self.theta
        return np.concatenate([lidar, [distance, heading_error]])

    # 충돌 감지
    def _check_collision(self):
        if self.y <= 0 or self.y >= self.HEIGHT:
            return True
        for (ox, oy) in self.obstacles:
            if np.hypot(self.x - ox, self.y - oy) < 15:
                return True
        return False

    # 환경 초기화
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x, self.y = 50.0, self.HEIGHT / 2
        self.theta = np.random.uniform(-0.1, 0.1)
        self.goal = np.array([self.WIDTH - 50, self.HEIGHT / 2])
        self.path = []
        self.log_data = []  # 매번 초기화

        # 장애물 랜덤 생성
        self.obstacles = []
        for _ in range(6):
            ox = random.randint(200, 800)
            oy = random.randint(40, 160)
            self.obstacles.append((ox, oy))

        observation = self._get_state()
        info = {}
        return observation, info

    # 시뮬레이션 진행 (한 스텝)
    def step(self, action):
        v = 4.0  # 속도
        w = 0.1 if action == 2 else (-0.1 if action == 0 else 0.0)

        self.x += math.cos(self.theta) * v
        self.y += math.sin(self.theta) * v
        self.theta += w

        lidar = self._get_lidar_scan()
        dx, dy = self.goal - np.array([self.x, self.y])
        distance = np.hypot(dx, dy)
        heading_error = math.atan2(dy, dx) - self.theta

        reward = -0.1
        terminated = False
        truncated = False

        if self._check_collision():
            reward -= 100
            terminated = True
        elif distance < 20:
            reward += 100
            terminated = True
        else:
            reward += (500 - distance) * 0.001
            reward -= abs(heading_error) * 0.05

        # 로그 데이터 저장
        self.log_data.append([
            self.x, self.y, self.theta, v, w, distance, heading_error, *lidar
        ])

        self.path.append((self.x, self.y))
        if self.render_mode:
            self.render()

        info = {}
        return self._get_state(), reward, terminated, truncated, info

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((230, 245, 255))
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, self.WIDTH, 5))
        pygame.draw.rect(self.screen, (0, 0, 0), (0, self.HEIGHT - 5, self.WIDTH, 5))
        pygame.draw.circle(self.screen, (255, 0, 0), (int(self.goal[0]), int(self.goal[1])), 10)

        for (ox, oy) in self.obstacles:
            pygame.draw.circle(self.screen, (128, 0, 128), (int(ox), int(oy)), 12)

        for p in self.path:
            pygame.draw.circle(self.screen, (0, 0, 255), (int(p[0]), int(p[1])), 2)
        pygame.draw.circle(self.screen, (0, 200, 0), (int(self.x), int(self.y)), 6)

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if len(self.log_data) > 0:
            with open("boat_run_log.csv", "w", newline="") as f:
                writer = csv.writer(f)
                header = ["x", "y", "theta", "v", "omega", "distance", "heading_error"] + [f"lidar_{i}" for i in range(9)]
                writer.writerow(header)
                writer.writerows(self.log_data)
        if self.render_mode:
            pygame.quit()
