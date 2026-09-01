import pygame
import numpy as np
import math

class EnvRenderer:
    def __init__(self, env):
        self.env = env
        self.pov_surf = pygame.Surface((320, 220), pygame.SRCALPHA)
        self.cam_surf = pygame.Surface((320, 220), pygame.SRCALPHA)
        self.real_cam_surf = pygame.Surface((320, 220), pygame.SRCALPHA)
        self.safety_surf = pygame.Surface((120, 120), pygame.SRCALPHA)
        self.hud_surf = pygame.Surface((210, 110), pygame.SRCALPHA)
        self.font = pygame.font.SysFont(None, 24)
        self.bold_font = pygame.font.SysFont(None, 26, bold=True)
        self.small_font = pygame.font.SysFont(None, 18)

    def render(self, hits):
        env = self.env
        # 1. 심해 그라데이션 해수면 배경
        env.screen.fill((16, 68, 115))
        
        # 해양 차트 미세 그리드 격자 (Sonar Depth Grid)
        for gx in range(0, env.w, 100):
            pygame.draw.line(env.screen, (22, 82, 132), (gx, 0), (gx, env.sim_h), 1)
        for gy in range(0, env.sim_h, 100):
            pygame.draw.line(env.screen, (22, 82, 132), (0, gy), (env.w, gy), 1)

        # 역동적인 다층 수면 파도 물결 (Dynamic Ocean Waves & Caustics)
        wave_t = env.frame * 0.04
        for i in range(15, env.sim_h, 45):
            for j in range(15, env.w, 65):
                wx = j + math.cos(wave_t + i * 0.08) * 14
                wy = i + math.sin(wave_t * 0.8 + j * 0.08) * 7
                w_len = 22 + math.sin(wave_t + j * 0.05) * 8
                # 파도 하이라이트
                pygame.draw.line(env.screen, (32, 112, 172), (int(wx), int(wy)), (int(wx + w_len), int(wy)), 2)
                pygame.draw.line(env.screen, (70, 160, 225), (int(wx + 4), int(wy - 1)), (int(wx + w_len * 0.6), int(wy - 1)), 1)

        bx, by = env.boat_pos
        h = env.boat_heading
        ch, sh = math.cos(h), math.sin(h)
        
        # 2. 360도 라이다 범위
        if env.show_lidar_range:
            pygame.draw.circle(env.screen, (0, 180, 100), (int(bx), int(by)), int(env.lidar_range), 1)
            for ang in env.rel_angles:
                ray_ang = h + ang
                rx = bx + math.cos(ray_ang) * env.lidar_range
                ry = by + math.sin(ray_ang) * env.lidar_range
                pygame.draw.line(env.screen, (0, 100, 60), (int(bx), int(by)), (int(rx), int(ry)), 1)

        # 3. 항적 및 거품 웨이크 (Propeller Foam & Twin Wakes)
        env.wake_surf.fill((0, 0, 0, 0))
        for w in env.wakes:
            w[2] += 0.8
            w[3] -= 3.2
            if w[3] > 0:
                # 외곽 물거품
                pygame.draw.circle(env.wake_surf, (220, 240, 255, int(w[3] * 0.5)), (int(w[0]), int(w[1])), int(w[2]))
                # 중심 백색 기포
                if w[2] > 3:
                    pygame.draw.circle(env.wake_surf, (255, 255, 255, int(w[3])), (int(w[0]), int(w[1])), int(w[2] * 0.45))
        env.wakes = [w for w in env.wakes if w[3] > 0]
        
        env.screen.blit(env.wake_surf, (0, 0))
        env.screen.blit(env.trail, (0, 0))
        
        # 4. 해상 장애물 (부표 링 파도 + 3D 엠보싱 구체)
        for ox, oy, r in env.dynamic_obstacles:
            # 부표 주변 수면 반사 링 파도
            ripple_r = r + 4 + math.sin(env.frame * 0.08 + ox * 0.1) * 3
            pygame.draw.circle(env.screen, (25, 95, 150), (int(ox), int(oy)), int(ripple_r), 1)
            
            # 부표 그림자
            pygame.draw.circle(env.screen, (10, 40, 75, 160), (int(ox + 4), int(oy + 4)), int(r + 1))
            # 부표 바디 (해양 안전 오렌지)
            pygame.draw.circle(env.screen, (225, 55, 30), (int(ox), int(oy)), int(r))
            pygame.draw.circle(env.screen, (255, 95, 60), (int(ox - 2), int(oy - 2)), int(r * 0.75))
            # 부표 상단 반사광 및 발광 비콘
            pygame.draw.circle(env.screen, (255, 220, 190), (int(ox - 3), int(oy - 3)), int(r * 0.38))
            pygame.draw.circle(env.screen, (255, 240, 80), (int(ox), int(oy)), 3)
            
        env.occ_surf.fill((0, 0, 0, 0))
        occ = np.where(env.grid >= 3)
        for gy, gx in zip(occ[0], occ[1]):
            x = gx * 4
            y = gy * 4
            pygame.draw.rect(env.occ_surf, (220, 50, 50, 60), (x, y, 4, 4))
        env.screen.blit(env.occ_surf, (0, 0))
            
        if env.show_lidar:
            for p in hits:
                if p is not None:
                    pygame.draw.circle(env.screen, (255, 255, 0), (int(p[0]), int(p[1])), 2)

        # Safety Envelope (안전 반경 오버레이)
        self.safety_surf.fill((0, 0, 0, 0))
        safety_r = int(env.boat_radius + 18)
        em = getattr(env, 'emergency_mode', False)
        safety_color = (255, 50, 50, 45) if em else (0, 220, 140, 22)
        pygame.draw.circle(self.safety_surf, safety_color, (60, 60), safety_r)
        pygame.draw.circle(self.safety_surf, (255, 60, 60, 120) if em else (0, 240, 160, 60), (60, 60), safety_r, 1)
        env.screen.blit(self.safety_surf, (int(bx - 60), int(by - 60)))
                
        # 목표점: 해양 항로 비콘 (Nautical Beacon Target)
        pulse = math.sin(env.frame * 0.09) * 4.5
        pygame.draw.circle(env.screen, (0, 240, 100, 40), (int(env.target[0]), int(env.target[1])), int(20 + pulse), 1)
        pygame.draw.circle(env.screen, (0, 230, 90, 75), (int(env.target[0]), int(env.target[1])), int(14 + pulse * 0.5))
        pygame.draw.circle(env.screen, (20, 245, 80), (int(env.target[0]), int(env.target[1])), 10)
        pygame.draw.circle(env.screen, (255, 255, 255), (int(env.target[0]), int(env.target[1])), 5)
        # 타겟 십자선 (Compass Crosshair)
        tx, ty = int(env.target[0]), int(env.target[1])
        pygame.draw.line(env.screen, (255, 255, 255, 180), (tx - 16, ty), (tx + 16, ty), 1)
        pygame.draw.line(env.screen, (255, 255, 255, 180), (tx, ty - 16), (tx, ty + 16), 1)
        
        env.screen.blit(env.path_surf, (0, 0))
        
        # 5. 경로 및 타겟 점
        if env.show_path2:
            if env.next_wp is not None:
                nwp = env.next_wp
                pygame.draw.line(env.screen, (255, 140, 0), (int(nwp["c1"][0]), int(nwp["c1"][1])), (int(nwp["c2"][0]), int(nwp["c2"][1])), 3)
                pygame.draw.circle(env.screen, (200, 100, 255, 100), (int(nwp["pos"][0]), int(nwp["pos"][1])), 8)
                pygame.draw.circle(env.screen, (200, 100, 255), (int(nwp["pos"][0]), int(nwp["pos"][1])), 3)

            if env.next_bezier_path is not None:
                pts = [(int(x), int(y)) for x, y in env.next_bezier_path]
                if len(pts) > 1:
                    pygame.draw.lines(env.screen, (255, 200, 50), False, pts, 3)

            if env.next_pursuit_target is not None:
                px_nt, py_nt = env.next_pursuit_target
                pygame.draw.circle(env.screen, (255, 255, 255), (int(px_nt), int(py_nt)), 8, 2)
                pygame.draw.circle(env.screen, (255, 150, 50), (int(px_nt), int(py_nt)), 4)

        if env.show_path1:
            if env.current_wp is not None:
                wp = env.current_wp
                pygame.draw.line(env.screen, (0, 255, 200), (int(wp["c1"][0]), int(wp["c1"][1])), (int(wp["c2"][0]), int(wp["c2"][1])), 4)
                pygame.draw.circle(env.screen, (0, 255, 255, 100), (int(wp["pos"][0]), int(wp["pos"][1])), 10)
                pygame.draw.circle(env.screen, (0, 255, 255), (int(wp["pos"][0]), int(wp["pos"][1])), 4)
                             
            if env.bezier_path is not None:
                pts = [(int(x), int(y)) for x, y in env.bezier_path]
                if len(pts) > 1:
                    pygame.draw.lines(env.screen, (50, 210, 255), False, pts, 4)

            if env.pursuit_target is not None:
                px_t, py_t = env.pursuit_target
                pygame.draw.circle(env.screen, (255, 255, 255), (int(px_t), int(py_t)), 10, 2)
                pygame.draw.circle(env.screen, (255, 50, 150), (int(px_t), int(py_t)), 5)

        # 6. 선박 형상 정밀 렌더링 (카타마란 + 항해등 + 듀얼 모터 + 라이다 회전)
        self._draw_boat_hull(bx, by, ch, sh)

        # 7. 하단 대시보드 UI
        self._draw_dashboard(hits)

        # 8. 실시간 텔레메트리 HUD
        self._draw_telemetry()

        pygame.display.flip()

    def _draw_boat_hull(self, bx, by, ch, sh):
        env = self.env
        GAP = 11; L = 84; W = 16
        left_center = (bx - sh*GAP, by + ch*GAP)
        right_center = (bx + sh*GAP, by - ch*GAP)
        
        hull_local = [
            (L*0.50, 0), (L*0.12, W),
            (-L*0.28, W*0.85), (-L*0.48, W*0.6),
            (-L*0.50, 0), (-L*0.48, -W*0.6),
            (-L*0.28, -W*0.85), (L*0.12, -W)
        ]
        
        def TR(c, px_l, py_l):
            return int(c[0] + px_l*ch - py_l*sh), int(c[1] + px_l*sh + py_l*ch)
            
        left_h = [TR(left_center, p[0], p[1]) for p in hull_local]
        right_h = [TR(right_center, p[0], p[1]) for p in hull_local]
        
        # 선체 하부 앰비언트 수중 그림자
        env.shadow_surf.fill((0, 0, 0, 0))
        shadow_offset = 7
        left_shadow = [(p[0]+shadow_offset, p[1]+shadow_offset) for p in left_h]
        right_shadow = [(p[0]+shadow_offset, p[1]+shadow_offset) for p in right_h]
        pygame.draw.polygon(env.shadow_surf, (8, 35, 65, 140), left_shadow)
        pygame.draw.polygon(env.shadow_surf, (8, 35, 65, 140), right_shadow)
        env.screen.blit(env.shadow_surf, (0, 0))

        # 좌/우 선체 (Navy Metallic Hull)
        pygame.draw.polygon(env.screen, (28, 75, 190), left_h)
        pygame.draw.polygon(env.screen, (15, 45, 120), left_h, 2)
        pygame.draw.polygon(env.screen, (28, 75, 190), right_h)
        pygame.draw.polygon(env.screen, (15, 45, 120), right_h, 2)

        # 좌우 선체 상단 데크 라인 (Highlight Streaks)
        left_deck_line = [TR(left_center, L*0.35, 0), TR(left_center, -L*0.35, 0)]
        right_deck_line = [TR(right_center, L*0.35, 0), TR(right_center, -L*0.35, 0)]
        pygame.draw.line(env.screen, (90, 160, 245), left_deck_line[0], left_deck_line[1], 2)
        pygame.draw.line(env.screen, (90, 160, 245), right_deck_line[0], right_deck_line[1], 2)

        # 중앙 연결 브릿지 데크 (Carbon Composite Bridge Deck)
        deck_corners = [
            TR((bx, by), L*0.25, -GAP*0.85),
            TR((bx, by), L*0.25, GAP*0.85),
            TR((bx, by), -L*0.35, GAP*0.85),
            TR((bx, by), -L*0.35, -GAP*0.85)
        ]
        pygame.draw.polygon(env.screen, (185, 195, 205), deck_corners)
        pygame.draw.polygon(env.screen, (60, 70, 80), deck_corners, 1)

        # 캐빈 조종실 (White Cabin Pod)
        cabin_corners = [
            TR((bx, by), L*0.16, -GAP*0.55),
            TR((bx, by), L*0.16, GAP*0.55),
            TR((bx, by), -L*0.16, GAP*0.55),
            TR((bx, by), -L*0.16, -GAP*0.55)
        ]
        pygame.draw.polygon(env.screen, (248, 250, 255), cabin_corners)
        pygame.draw.polygon(env.screen, (40, 50, 60), cabin_corners, 1)

        # 틴팅 전면 윈드실드 유리창 (Tinted Aero Glass)
        windshield = [
            TR((bx, by), L*0.13, -GAP*0.42),
            TR((bx, by), L*0.13, GAP*0.42),
            TR((bx, by), L*0.04, GAP*0.42),
            TR((bx, by), L*0.04, -GAP*0.42)
        ]
        pygame.draw.polygon(env.screen, (30, 180, 255), windshield)
        pygame.draw.line(env.screen, (200, 245, 255), TR((bx, by), L*0.12, -GAP*0.3), TR((bx, by), L*0.06, GAP*0.3), 1)

        # 해상 국제 항해등 (Navigation Lights: 좌현 적색 Port, 우현 녹색 Starboard)
        port_light = TR(left_center, L*0.42, W*0.3)
        stbd_light = TR(right_center, L*0.42, -W*0.3)
        stern_light = TR((bx, by), -L*0.36, 0)
        
        # 좌현 적색등 (Port Red)
        pygame.draw.circle(env.screen, (255, 30, 30), port_light, 3)
        pygame.draw.circle(env.screen, (255, 100, 100), port_light, 1)
        # 우현 녹색등 (Starboard Green)
        pygame.draw.circle(env.screen, (30, 255, 80), stbd_light, 3)
        pygame.draw.circle(env.screen, (150, 255, 180), stbd_light, 1)
        # 선미 백색등 (Stern White)
        pygame.draw.circle(env.screen, (255, 255, 255), stern_light, 2)

        # 듀얼 아웃보드 모터 추진기 (Dual Outboard Thrusters)
        left_motor = TR(left_center, -L*0.52, 0)
        right_motor = TR(right_center, -L*0.52, 0)
        pygame.draw.rect(env.screen, (35, 35, 40), (left_motor[0]-3, left_motor[1]-3, 6, 6))
        pygame.draw.rect(env.screen, (35, 35, 40), (right_motor[0]-3, right_motor[1]-3, 6, 6))

        # 중앙 회전식 라이다 센서 돔 (Rotating LiDAR Sensor Pod)
        Lidar_pos = TR((bx, by), -L*0.05, 0)
        pygame.draw.circle(env.screen, (40, 45, 55), Lidar_pos, 6)
        pygame.draw.circle(env.screen, (255, 215, 30), Lidar_pos, 3)
        # 라이다 스캔 펄스 점
        scan_ang = env.frame * 0.35
        sp_x = int(Lidar_pos[0] + math.cos(scan_ang) * 5)
        sp_y = int(Lidar_pos[1] + math.sin(scan_ang) * 5)
        pygame.draw.circle(env.screen, (0, 255, 200), (sp_x, sp_y), 2)

    def _draw_dashboard(self, hits):
        env = self.env
        bx, by = env.boat_pos
        h = env.boat_heading
        ch, sh = math.cos(h), math.sin(h)

        pygame.draw.rect(env.screen, (15, 35, 60), (0, env.sim_h, env.w, env.h - env.sim_h))
        pygame.draw.line(env.screen, (0, 180, 255), (0, env.sim_h), (env.w, env.sim_h), 3)

        # 체크박스 렌더링
        pygame.draw.rect(env.screen, (255, 255, 255), env.cb1_rect, 2)
        if env.show_path1: pygame.draw.rect(env.screen, (0, 255, 200), env.cb1_rect.inflate(-6, -6))
        env.screen.blit(self.font.render("Show 1st Path Set", True, (255, 255, 255)), (70, 672))

        pygame.draw.rect(env.screen, (255, 255, 255), env.cb2_rect, 2)
        if env.show_path2: pygame.draw.rect(env.screen, (255, 200, 50), env.cb2_rect.inflate(-6, -6))
        env.screen.blit(self.font.render("Show 2nd Path Set", True, (255, 255, 255)), (70, 712))

        pygame.draw.rect(env.screen, (255, 255, 255), env.cb3_rect, 2)
        if env.show_lidar: pygame.draw.rect(env.screen, (255, 255, 0), env.cb3_rect.inflate(-6, -6))
        env.screen.blit(self.font.render("Show LiDAR Hits", True, (255, 255, 255)), (70, 752))

        pygame.draw.rect(env.screen, (255, 255, 255), env.cb4_rect, 2)
        if env.show_lidar_range: pygame.draw.rect(env.screen, (0, 180, 100), env.cb4_rect.inflate(-6, -6))
        env.screen.blit(self.font.render("Show LiDAR Range", True, (255, 255, 255)), (70, 792))

        # --- 1. 180도 전방 확대 LiDAR View (2D) ---
        pov_w, pov_h = 320, 220
        self.pov_surf.fill((10, 25, 45, 240))
        pygame.draw.rect(self.pov_surf, (0, 180, 255), (0, 0, pov_w, pov_h), 2)
        
        pcx, pcy = pov_w // 2, pov_h - 25
        f_vec = np.array([ch, sh])
        r_vec = np.array([-sh, ch])
        
        scale_r = 0.55

        # 180도 전방 부채꼴 가이드라인 및 방위각 레이더 그리드
        angles_deg = [-90, -60, -30, 0, 30, 60, 90]
        for deg in angles_deg:
            rad = math.radians(deg)
            rx = pcx + math.sin(rad) * (env.lidar_range * scale_r)
            ry = pcy - math.cos(rad) * (env.lidar_range * scale_r)
            pygame.draw.line(self.pov_surf, (0, 70, 110), (pcx, pcy), (int(rx), int(ry)), 1)
            
            display_deg = deg + 90
            txt_ang = self.small_font.render(f"{display_deg}°", True, (0, 140, 200))
            tx_off = -12 if deg < 0 else (-6 if deg == 0 else 2)
            ty_off = -12 if ry < pcy else 2
            self.pov_surf.blit(txt_ang, (int(rx) + tx_off, int(ry) + ty_off))

        # 동심원 스케일 서클 (50px = 1m 기준: 2m, 4m, 6m)
        for dist in [100, 200, 300]:
            r_pixel = int(dist * scale_r)
            rect = pygame.Rect(pcx - r_pixel, pcy - r_pixel, r_pixel * 2, r_pixel * 2)
            pygame.draw.arc(self.pov_surf, (0, 90, 140), rect, 0, math.pi, 1)
            lbl = self.small_font.render(f"{dist // 50}m", True, (0, 120, 170))
            self.pov_surf.blit(lbl, (pcx + 4, pcy - r_pixel - 10))

        # 180도 스캔 레이 라인
        if env.show_lidar_range:
            for ang in env.rel_angles:
                if -math.pi/2 <= ang <= math.pi/2:
                    rx = pcx + math.sin(ang) * (env.lidar_range * scale_r)
                    ry = pcy - math.cos(ang) * (env.lidar_range * scale_r)
                    pygame.draw.line(self.pov_surf, (0, 110, 60), (pcx, pcy), (int(rx), int(ry)), 1)

        # 라이다 히트 포인트 렌더링
        if env.show_lidar:
            for hp in hits:
                if hp is not None:
                    hdx = hp[0] - bx; hdy = hp[1] - by
                    hlf = hdx * f_vec[0] + hdy * f_vec[1]
                    hlr = hdx * r_vec[0] + hdy * r_vec[1]
                    if hlf >= -10:
                        pygame.draw.circle(self.pov_surf, (255, 255, 0), (int(pcx + hlr * scale_r), int(pcy - hlf * scale_r)), 2)

        # --- 목적지 인디케이터 & 테두리 트래킹 컴퍼스 ---
        dx_t = env.target[0] - bx; dy_t = env.target[1] - by
        lf_t = dx_t * f_vec[0] + dy_t * f_vec[1]
        lr_t = dx_t * r_vec[0] + dy_t * r_vec[1]
        
        tx_p = pcx + lr_t * scale_r
        ty_p = pcy - lf_t * scale_r
        
        margin = 0
        dist_total_m = math.hypot(dx_t, dy_t) / 50.0
        
        if margin <= tx_p <= pov_w - margin and margin <= ty_p <= pov_h - margin:
            pygame.draw.circle(self.pov_surf, (20, 250, 80), (int(tx_p), int(ty_p)), 7)
            pygame.draw.circle(self.pov_surf, (255, 255, 255), (int(tx_p), int(ty_p)), 3)
        else:
            dir_x = tx_p - pcx
            dir_y = ty_p - pcy
            
            t_candidates = []
            if dir_x < 0: t_candidates.append((margin - pcx) / dir_x)
            elif dir_x > 0: t_candidates.append(((pov_w - margin) - pcx) / dir_x)
            
            if dir_y < 0: t_candidates.append((margin - pcy) / dir_y)
            elif dir_y > 0: t_candidates.append(((pov_h - margin) - pcy) / dir_y)
            
            valid_t = [t for t in t_candidates if t > 0]
            if valid_t:
                t_edge = min(valid_t)
                edge_x = int(pcx + t_edge * dir_x)
                edge_y = int(pcy + t_edge * dir_y)
                
                pygame.draw.line(self.pov_surf, (20, 220, 80), (pcx, pcy), (edge_x, edge_y), 1)
                pygame.draw.circle(self.pov_surf, (20, 250, 80), (edge_x, edge_y), 6)
                pygame.draw.circle(self.pov_surf, (255, 255, 255), (edge_x, edge_y), 2)
                
                dist_txt = self.small_font.render(f"{dist_total_m:.0f}m", True, (20, 250, 80))
                lbl_x = max(10, min(edge_x - 12, pov_w - 40))
                lbl_y = max(10, min(edge_y - 12, pov_h - 18))
                self.pov_surf.blit(dist_txt, (lbl_x, lbl_y))

        # 내 선체 형상
        pygame.draw.circle(self.pov_surf, (20, 60, 180), (pcx, pcy), int(env.boat_radius * scale_r))
        pygame.draw.line(self.pov_surf, (255, 255, 255), (pcx, pcy), (pcx, pcy - 16), 2)
        
        txt_surf = self.font.render("LiDAR View", True, (255, 255, 255))
        self.pov_surf.blit(txt_surf, (10, pov_h - txt_surf.get_height() - 5))
        env.screen.blit(self.pov_surf, (350, env.sim_h + 35))

        # --- 2. 180도 라이다 각도 세로 게이지 뷰 (LiDAR Gauge View) ---
        cam_w, cam_h = 320, 220
        self.cam_surf.fill((10, 20, 35, 240))
        pygame.draw.rect(self.cam_surf, (0, 180, 255), (0, 0, cam_w, cam_h), 2)

        n_slices = 180
        slice_angles = np.linspace(-np.pi/2, np.pi/2, n_slices)

        # 180개 각도 세로 직사각형 게이지 렌더링
        for i in range(n_slices):
            ang = slice_angles[i]
            idx = int((ang + np.pi) / (2 * np.pi) * len(env.rel_angles)) % len(env.rel_angles)
            hp = hits[idx] if idx < len(hits) else None
            
            if hp is not None:
                hdx = hp[0] - bx
                hdy = hp[1] - by
                d = math.hypot(hdx, hdy)
            else:
                d = env.lidar_range

            x1 = int(i * cam_w / n_slices)
            x2 = int((i + 1) * cam_w / n_slices)
            w_s = max(1, x2 - x1)

            if d < env.lidar_range:
                if d < 70:
                    color = (230, 60, 50)
                elif d < 140:
                    color = (240, 160, 40)
                elif d < 220:
                    color = (210, 210, 50)
                else:
                    color = (40, 170, 160)
                
                pygame.draw.rect(self.cam_surf, color, (x1, 2, w_s, cam_h - 4))

        # 각도 보조 구분선
        for deg in [-60, -30, 0, 30, 60]:
            s_idx = int((deg + 90) / 180.0 * n_slices)
            gx = int(s_idx * cam_w / n_slices)
            pygame.draw.line(self.cam_surf, (255, 255, 255, 70), (gx, 0), (gx, cam_h), 1)

        # 웨이포인트 및 최종 목표 지점 수직 오버레이 신호선
        marker_objs = []
        if env.show_path1 and env.current_wp is not None:
            dx_w = env.current_wp["pos"][0] - bx; dy_w = env.current_wp["pos"][1] - by
            lf_w = dx_w * f_vec[0] + dy_w * f_vec[1]; lr_w = dx_w * r_vec[0] + dy_w * r_vec[1]
            marker_objs.append(('wp1', lf_w, lr_w))

        if env.show_path2 and env.next_wp is not None:
            dx_w2 = env.next_wp["pos"][0] - bx; dy_w2 = env.next_wp["pos"][1] - by
            lf_w2 = dx_w2 * f_vec[0] + dy_w2 * f_vec[1]; lr_w2 = dx_w2 * r_vec[0] + dy_w2 * r_vec[1]
            marker_objs.append(('wp2', lf_w2, lr_w2))

        marker_objs.append(('target', lf_t, lr_t))

        for obj_type, lf, lr in marker_objs:
            ang_obj = math.atan2(lr, lf)
            if -math.pi/2 <= ang_obj <= math.pi/2:
                s_idx = int((ang_obj + math.pi/2) / math.pi * n_slices)
                mx = int(s_idx * cam_w / n_slices)
                
                if obj_type == 'wp1':
                    pygame.draw.line(self.cam_surf, (0, 255, 220), (mx, 0), (mx, cam_h), 2)
                    pygame.draw.circle(self.cam_surf, (0, 255, 220), (mx, 25), 6)
                    pygame.draw.circle(self.cam_surf, (255, 255, 255), (mx, 25), 2)
                elif obj_type == 'wp2':
                    pygame.draw.line(self.cam_surf, (200, 100, 255), (mx, 0), (mx, cam_h), 2)
                    pygame.draw.circle(self.cam_surf, (200, 100, 255), (mx, 45), 6)
                    pygame.draw.circle(self.cam_surf, (255, 255, 255), (mx, 45), 2)
                elif obj_type == 'target':
                    pygame.draw.line(self.cam_surf, (20, 250, 80), (mx, 0), (mx, cam_h), 3)
                    pygame.draw.circle(self.cam_surf, (20, 250, 80), (mx, 65), 7)
                    pygame.draw.circle(self.cam_surf, (255, 255, 255), (mx, 65), 3)

        self.cam_surf.blit(self.font.render("LiDAR Gauge View", True, (255, 255, 255)), (10, 8))
        env.screen.blit(self.cam_surf, (700, env.sim_h + 35))

        # --- 3. LiDAR Depth 1st-Person View (하단 렌더링: 해수면 배경 처리) ---
        real_w, real_h = 320, 220
        self.real_cam_surf.fill((10, 20, 35, 240))
        
        horizon_y = real_h // 2 + 10
        pygame.draw.rect(self.real_cam_surf, (15, 30, 55), (0, 0, real_w, horizon_y))
        
        # 하단 절반(수평선 아래) 기본 해수면 그라데이션 및 물결 패턴
        for y in range(horizon_y, real_h):
            ratio = (y - horizon_y) / float(real_h - horizon_y)
            r_sea = int(12 - ratio * 5)
            g_sea = int(45 + ratio * 35)
            b_sea = int(95 + ratio * 45)
            pygame.draw.line(self.real_cam_surf, (r_sea, g_sea, b_sea), (0, y), (real_w, y))

        for wy_off in [12, 28, 50, 78]:
            y_p = horizon_y + wy_off
            if y_p < real_h:
                pygame.draw.line(self.real_cam_surf, (25, 95, 155, 90), (0, y_p), (real_w, y_p), 1)

        pygame.draw.line(self.real_cam_surf, (0, 160, 220), (0, horizon_y), (real_w, horizon_y), 1)

        # 전방 180도를 180개 슬라이스로 분할하여 각 라이다 거리 막대 렌더링
        for i in range(n_slices):
            ang = slice_angles[i]
            idx = int((ang + np.pi) / (2 * np.pi) * len(env.rel_angles)) % len(env.rel_angles)
            hp = hits[idx] if idx < len(hits) else None
            
            if hp is not None:
                hdx = hp[0] - bx
                hdy = hp[1] - by
                d = math.hypot(hdx, hdy)
            else:
                d = env.lidar_range

            if d < env.lidar_range:
                bar_h = min(real_h - 10, int(11000.0 / max(d, 12.0)))
                x1 = int(i * real_w / n_slices)
                x2 = int((i + 1) * real_w / n_slices)
                w_s = max(1, x2 - x1)
                
                y_top = horizon_y - bar_h // 2
                
                if d < 70:
                    color = (230, 60, 50)
                elif d < 140:
                    color = (240, 160, 40)
                elif d < 220:
                    color = (210, 210, 50)
                else:
                    color = (40, 170, 160)
                
                pygame.draw.rect(self.real_cam_surf, color, (x1, y_top, w_s, bar_h))

        # 라이다 거릿값 막대 위에 오버레이되는 웨이포인트 및 최종 목표 지점 핀 마커
        overlay_objs = []
        if env.show_path1 and env.current_wp is not None:
            dx_w = env.current_wp["pos"][0] - bx; dy_w = env.current_wp["pos"][1] - by
            lf_w = dx_w * f_vec[0] + dy_w * f_vec[1]; lr_w = dx_w * r_vec[0] + dy_w * r_vec[1]
            if lf_w > 2.0: overlay_objs.append(('wp1', lf_w, lr_w))

        if env.show_path2 and env.next_wp is not None:
            dx_w2 = env.next_wp["pos"][0] - bx; dy_w2 = env.next_wp["pos"][1] - by
            lf_w2 = dx_w2 * f_vec[0] + dy_w2 * f_vec[1]; lr_w2 = dx_w2 * r_vec[0] + dy_w2 * r_vec[1]
            if lf_w2 > 2.0: overlay_objs.append(('wp2', lf_w2, lr_w2))

        if lf_t > 2.0:
            overlay_objs.append(('target', lf_t, lr_t))

        overlay_objs.sort(key=lambda item: item[1], reverse=True)

        for obj_type, lf, lr in overlay_objs:
            angle = math.atan2(lr, lf)
            if abs(angle) <= math.pi / 2:
                sx = int(real_w / 2 + (angle / (math.pi / 2)) * (real_w / 2))
                sy_base = int(horizon_y + (160.0 / max(lf, 10.0)) * 12)
                sy_base = min(sy_base, real_h - 10)
                
                scale_factor = 200.0 / max(lf, 10.0)
                pole_h = max(12, int(40 * scale_factor * 0.35))
                pole_y = sy_base - pole_h
                
                if obj_type == 'wp1':
                    pygame.draw.line(self.real_cam_surf, (0, 255, 220), (sx, sy_base), (sx, pole_y), 2)
                    pygame.draw.circle(self.real_cam_surf, (0, 255, 220), (sx, pole_y), 5)
                    pygame.draw.circle(self.real_cam_surf, (255, 255, 255), (sx, pole_y), 2)
                elif obj_type == 'wp2':
                    pygame.draw.line(self.real_cam_surf, (200, 100, 255), (sx, sy_base), (sx, pole_y), 2)
                    pygame.draw.circle(self.real_cam_surf, (200, 100, 255), (sx, pole_y), 5)
                    pygame.draw.circle(self.real_cam_surf, (255, 255, 255), (sx, pole_y), 2)
                elif obj_type == 'target':
                    pygame.draw.line(self.real_cam_surf, (20, 250, 80), (sx, sy_base), (sx, pole_y), 3)
                    pygame.draw.circle(self.real_cam_surf, (20, 250, 80), (sx, pole_y), 7)
                    pygame.draw.circle(self.real_cam_surf, (255, 255, 255), (sx, pole_y), 3)

        pygame.draw.rect(self.real_cam_surf, (130, 180, 220), (0, 0, real_w, real_h), 2)
        self.real_cam_surf.blit(self.font.render("LiDAR 1st View", True, (255, 255, 255)), (10, 10))
        env.screen.blit(self.real_cam_surf, (1050, env.sim_h + 35))

    def _draw_telemetry(self):
        """우상단 실시간 텔레메트리 HUD"""
        env = self.env
        hud_w, hud_h = 210, 110
        hud_x = env.w - hud_w - 15
        hud_y = 12
        
        hud_surf = self.hud_surf
        hud_surf.fill((10, 20, 40, 190))
        pygame.draw.rect(hud_surf, (0, 160, 230), (0, 0, hud_w, hud_h), 2)
        
        # 모드 표시
        em = getattr(env, 'emergency_mode', False)
        has_wp = env.current_wp is not None
        if em:
            mode_txt = self.bold_font.render("\u26a0 EMERGENCY", True, (255, 80, 60))
        elif has_wp:
            mode_txt = self.bold_font.render("\u25c8 GAP PASS", True, (0, 255, 220))
        else:
            mode_txt = self.bold_font.render("\u25b6 CRUISING", True, (50, 230, 120))
        hud_surf.blit(mode_txt, (10, 6))
        
        # 속도
        speed = float(np.linalg.norm(env.boat_vel))
        speed_knots = speed * 0.9
        spd_txt = self.small_font.render(f"Speed: {speed_knots:.1f} kt", True, (220, 235, 255))
        hud_surf.blit(spd_txt, (10, 32))
        
        # 속도 바
        bar_w = 125
        pygame.draw.rect(hud_surf, (30, 50, 70), (10, 48, bar_w, 7))
        fill_w = int(min(speed / 15.0, 1.0) * bar_w)
        bar_color = (255, 80, 60) if em else (0, 200, 100)
        pygame.draw.rect(hud_surf, bar_color, (10, 48, fill_w, 7))
        
        # 조타각
        steer_val = getattr(env, 'prev_steer', 0)
        steer_txt = self.small_font.render(f"Steer: {steer_val:+.2f}", True, (220, 235, 255))
        hud_surf.blit(steer_txt, (10, 60))
        
        # Heading
        hdg_deg = math.degrees(env.boat_heading) % 360
        hdg_txt = self.small_font.render(f"Heading: {hdg_deg:.0f}\u00b0", True, (220, 235, 255))
        hud_surf.blit(hdg_txt, (10, 76))
        
        # 목표 거리 (50px = 1m 기준 미터 단위 변환)
        d2t = float(np.linalg.norm(env.target - env.boat_pos))
        d2t_m = d2t / 50.0
        d2t_txt = self.small_font.render(f"Target: {d2t_m:.1f} m", True, (50, 230, 120))
        hud_surf.blit(d2t_txt, (10, 92))
        
        env.screen.blit(hud_surf, (hud_x, hud_y))