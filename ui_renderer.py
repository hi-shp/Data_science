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
        self.bezier_surf = pygame.Surface((190, 220), pygame.SRCALPHA)
        self.weights_surf = pygame.Surface((190, 220), pygame.SRCALPHA)
        self.font = pygame.font.SysFont(None, 24)
        self.bold_font = pygame.font.SysFont(None, 26, bold=True)
        self.small_font = pygame.font.SysFont(None, 18)
        self.micro_font = pygame.font.SysFont(None, 15)
        self.curv_buffer = None
        self.curv_y_max = 0.08
        self.smooth_path_m = 0.0

    def render(self, hits):
        env = self.env
        # 1. 밝고 맑은 마린 오션 수면 배경 (Brighter Clean Ocean)
        env.screen.fill((40, 118, 178))
        
        # 아주 은은하고 자연스러운 해양 잔물결 파도 (Gentle Natural Ocean Swell Waves)
        wave_t = env.frame * 0.016
        for i in range(25, env.sim_h, 60):
            for j in range(25, env.w, 80):
                wx = j + math.cos(wave_t + i * 0.025 + j * 0.01) * 9
                wy = i + math.sin(wave_t * 0.7 + j * 0.025) * 5
                w_len = 16 + math.sin(wave_t + j * 0.02) * 6
                # 물결 본선 (부드러운 오션 하이라이트)
                pygame.draw.line(env.screen, (55, 134, 196), (int(wx), int(wy)), (int(wx + w_len), int(wy)), 1)
                # 아주 은은한 미세 물결 크레스트 포인트 (Soft Crest Tip)
                if (i + j) % 160 == 0:
                    pygame.draw.circle(env.screen, (210, 235, 255), (int(wx + w_len * 0.5), int(wy - 1)), 1)

        bx, by = env.boat_pos
        h = env.boat_heading
        ch, sh = math.cos(h), math.sin(h)
        
        # 2. 360도 라이다 범위 (저채도 소프트 에메랄드 / 세이지 그린)
        if env.show_lidar_range:
            pygame.draw.circle(env.screen, (80, 175, 140), (int(bx), int(by)), int(env.lidar_range), 1)
            for ang in env.rel_angles:
                ray_ang = h + ang
                rx = bx + math.cos(ray_ang) * env.lidar_range
                ry = by + math.sin(ray_ang) * env.lidar_range
                pygame.draw.line(env.screen, (55, 115, 90), (int(bx), int(by)), (int(rx), int(ry)), 1)

        # 3. 실제 선박 유체역학 항적 웨이크 + 장애물 반사/산란 미세 거품 (Realistic Wakes & Scattering Bubbles)
        env.wake_surf.fill((0, 0, 0, 0))

        for w in env.wakes:
            # w: [x, y, radius, alpha, vx, vy]
            if len(w) >= 6:
                w[0] += w[4] * 0.80
                w[1] += w[5] * 0.80
                w[4] *= 0.93
                w[5] *= 0.93
            w[2] += 1.15  # 웅장하고 풍성한 선미 거품 확장
            w[3] -= 2.4   # 긴 항적 지속성
            if w[3] > 0:
                # 외곽 확장 파도 크레스트 (Wave Crest)
                pygame.draw.circle(env.wake_surf, (220, 240, 255, int(w[3] * 0.42)), (int(w[0]), int(w[1])), int(w[2]))
                # 내부 백색 기포 난류 (Aerated Foam Core)
                if w[2] > 1.8:
                    pygame.draw.circle(env.wake_surf, (255, 255, 255, int(w[3] * 0.72)), (int(w[0]), int(w[1])), int(w[2] * 0.52))
        env.wakes = [w for w in env.wakes if w[3] > 0]
        
        # 장애물 충돌 반사 및 부표 들썩임 시 퍼지는 원형 백색 구름 파도
        if hasattr(env, 'reflected_wakes'):
            for rw in env.reflected_wakes:
                if len(rw) == 4:
                    rw[2] += 0.38
                    rw[3] -= 2.2
                    if rw[3] > 0:
                        pygame.draw.circle(env.wake_surf, (225, 242, 255, int(rw[3] * 0.48)), (int(rw[0]), int(rw[1])), int(rw[2]), 2)
                elif len(rw) >= 6:
                    rw[0] += rw[4]; rw[1] += rw[5]
                    rw[4] *= 0.88; rw[5] *= 0.88
                    rw[3] -= 6.0
                    if rw[3] > 0:
                        pygame.draw.circle(env.wake_surf, (255, 255, 255, int(rw[3] * 0.8)), (int(rw[0]), int(rw[1])), 1)
            env.reflected_wakes = [rw for rw in env.reflected_wakes if rw[3] > 0]
            
        # 장애물 부표 위치의 파도/구름을 완전히 지워 가림 처리 (Obstacle Clean Masking)
        for ox, oy, r in env.dynamic_obstacles:
            pygame.draw.circle(env.wake_surf, (0, 0, 0, 0), (int(ox), int(oy)), int(r + 0.5))
        
        env.screen.blit(env.wake_surf, (0, 0))
        env.screen.blit(env.trail, (0, 0))
        
        # 4. 해상 장애물 (자연스러운 수중 그림자 + 선명한 2D 부표)
        for ox, oy, r in env.dynamic_obstacles:
            # 부표 수중 앰비언트 그림자 (Realistic Underwater Shadow)
            pygame.draw.circle(env.screen, (10, 42, 75, 140), (int(ox + 4), int(oy + 4)), int(r + 1))
            
            # 선명한 2D 부표 바디 (중심부 쨍한 순백색 하이라이트)
            pygame.draw.circle(env.screen, (210, 45, 30), (int(ox), int(oy)), int(r))
            pygame.draw.circle(env.screen, (245, 75, 50), (int(ox - 1), int(oy - 1)), int(r * 0.76))
            pygame.draw.circle(env.screen, (255, 255, 255), (int(ox - 1), int(oy - 1)), int(r * 0.40))
            pygame.draw.circle(env.screen, (255, 255, 255), (int(ox), int(oy)), int(r * 0.20))
            
        occ_y, occ_x = np.where(env.grid >= 3)
        if len(occ_x) > 0:
            env.occ_surf.fill((0, 0, 0, 0))
            for gx, gy in zip(occ_x, occ_y):
                pygame.draw.rect(env.occ_surf, (220, 50, 50, 60), (gx * 4, gy * 4, 4, 4))
            env.screen.blit(env.occ_surf, (0, 0))
            
        if env.show_lidar:
            for p in hits:
                if p is not None:
                    pygame.draw.circle(env.screen, (225, 220, 130), (int(p[0]), int(p[1])), 2)

        # Safety Envelope (8acbd56 버전의 소프트 반투명 안전 원, 120x120 로컬 버퍼)
        self.safety_surf.fill((0, 0, 0, 0))
        safety_r = int(env.boat_radius + 18)
        em = getattr(env, 'emergency_mode', False)
        safety_color = (255, 60, 60, 40) if em else (0, 200, 120, 25)
        pygame.draw.circle(self.safety_surf, safety_color, (60, 60), safety_r)
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
        # 5. 경로 및 타겟 점
        # 5. 1차 및 2차 경로 및 타겟 점 렌더링
        show_1st = getattr(env, 'show_1st_path', getattr(env, 'show_paths', True))
        show_2nd = getattr(env, 'show_2nd_path', getattr(env, 'show_paths', True))

        if show_2nd:
            if env.next_wp is not None:
                nwp = env.next_wp
                if nwp.get("pair") != (-1, -1):
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

        if show_1st:
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

        # 6. 차순위 후보 웨이포인트 렌더링 (투명도 적용)
        if getattr(env, 'show_candidates', True) and getattr(env, 'candidate_wps', None):
            cand_surf = pygame.Surface((env.w, env.h), pygame.SRCALPHA)
            cand_colors = [(80, 210, 255, 130), (255, 180, 70, 120)]
            for rank_idx, cand in enumerate(env.candidate_wps[:2]):
                col = cand_colors[rank_idx % len(cand_colors)]
                c1, c2 = cand["c1"], cand["c2"]
                mid = cand["pos"]
                pygame.draw.line(cand_surf, col, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), 2)
                pygame.draw.circle(cand_surf, (col[0], col[1], col[2], 50), (int(mid[0]), int(mid[1])), 9)
                pygame.draw.circle(cand_surf, col, (int(mid[0]), int(mid[1])), 9, 2)
                pygame.draw.circle(cand_surf, (col[0], col[1], col[2], 210), (int(mid[0]), int(mid[1])), 3)
                
                txt_rank = self.small_font.render(f"#{rank_idx + 2}", True, (col[0], col[1], col[2]))
                cand_surf.blit(txt_rank, (int(mid[0]) + 10, int(mid[1]) - 8))
            env.screen.blit(cand_surf, (0, 0))

        # 6. 선박 형상 정밀 렌더링
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
        pygame.draw.polygon(env.shadow_surf, (8, 30, 55, 150), left_shadow)
        pygame.draw.polygon(env.shadow_surf, (8, 30, 55, 150), right_shadow)
        env.screen.blit(env.shadow_surf, (0, 0))

        # 좌/우 선체 (군함/실험선 건메탈 그레이 - Tone 1: Gunmetal Grey)
        pygame.draw.polygon(env.screen, (52, 60, 70), left_h)
        pygame.draw.polygon(env.screen, (28, 34, 40), left_h, 2)
        pygame.draw.polygon(env.screen, (52, 60, 70), right_h)
        pygame.draw.polygon(env.screen, (28, 34, 40), right_h, 2)

        # 좌우 선체 상단 하이라이트 스트립
        left_deck_line = [TR(left_center, L*0.35, 0), TR(left_center, -L*0.35, 0)]
        right_deck_line = [TR(right_center, L*0.35, 0), TR(right_center, -L*0.35, 0)]
        pygame.draw.line(env.screen, (85, 96, 108), left_deck_line[0], left_deck_line[1], 2)
        pygame.draw.line(env.screen, (85, 96, 108), right_deck_line[0], right_deck_line[1], 2)

        # 중앙 연결 브릿지 데크 (투톤 대비 - Tone 2: Crisp Platinum Deck)
        deck_corners = [
            TR((bx, by), L*0.25, -GAP*0.85),
            TR((bx, by), L*0.25, GAP*0.85),
            TR((bx, by), -L*0.35, GAP*0.85),
            TR((bx, by), -L*0.35, -GAP*0.85)
        ]
        pygame.draw.polygon(env.screen, (210, 218, 228), deck_corners)
        pygame.draw.polygon(env.screen, (90, 100, 112), deck_corners, 1)

        # 데크 중앙 미끄럼 방지 패드 라인
        deck_pad = [
            TR((bx, by), L*0.20, -GAP*0.65),
            TR((bx, by), L*0.20, GAP*0.65),
            TR((bx, by), -L*0.30, GAP*0.65),
            TR((bx, by), -L*0.30, -GAP*0.65)
        ]
        pygame.draw.polygon(env.screen, (165, 175, 188), deck_pad)

        # 캐빈 조종실 팟 (Stealth Tactical Cabin)
        cabin_corners = [
            TR((bx, by), L*0.16, -GAP*0.55),
            TR((bx, by), L*0.16, GAP*0.55),
            TR((bx, by), -L*0.16, GAP*0.55),
            TR((bx, by), -L*0.16, -GAP*0.55)
        ]
        pygame.draw.polygon(env.screen, (75, 84, 96), cabin_corners)
        pygame.draw.polygon(env.screen, (35, 42, 50), cabin_corners, 1)

        # 틴팅 전면 윈드실드 창문 (Tinted Marine Cockpit Glass)
        windshield = [
            TR((bx, by), L*0.13, -GAP*0.42),
            TR((bx, by), L*0.13, GAP*0.42),
            TR((bx, by), L*0.04, GAP*0.42),
            TR((bx, by), L*0.04, -GAP*0.42)
        ]
        pygame.draw.polygon(env.screen, (28, 105, 160), windshield)
        pygame.draw.line(env.screen, (180, 230, 255), TR((bx, by), L*0.12, -GAP*0.3), TR((bx, by), L*0.06, GAP*0.3), 1)

        # 후방 GPS 수신기 마스트 돔 & 통신 휩 안테나 (GPS Dome & Whip Antenna)
        gps_pos = TR((bx, by), -L*0.22, GAP*0.35)
        pygame.draw.circle(env.screen, (245, 248, 255), gps_pos, 4)
        pygame.draw.circle(env.screen, (60, 70, 80), gps_pos, 4, 1)
        # 휩 안테나
        ant_pos = TR((bx, by), -L*0.24, -GAP*0.35)
        pygame.draw.circle(env.screen, (30, 35, 40), ant_pos, 2)
        pygame.draw.line(env.screen, (200, 210, 220), ant_pos, (ant_pos[0]-1, ant_pos[1]-6), 2)

        # 선체 일체형 소형 T500 덕트 쓰러스터 (Integrated Compact T500 Thrusters)
        t_ch, t_sh = ch, sh
        t_nx, t_ny = -sh, ch
        d_len = 11; d_rad = 4.8
        
        for m_center in [left_center, right_center]:
            p_center = TR(m_center, -L*0.50, 0)
            
            # 덕트 노즐 모서리
            d_fl = (int(p_center[0] + (d_len*0.5)*t_ch - d_rad*t_nx), int(p_center[1] + (d_len*0.5)*t_sh - d_rad*t_ny))
            d_fr = (int(p_center[0] + (d_len*0.5)*t_ch + d_rad*t_nx), int(p_center[1] + (d_len*0.5)*t_sh + d_rad*t_ny))
            d_rr = (int(p_center[0] - (d_len*0.5)*t_ch + d_rad*t_nx), int(p_center[1] - (d_len*0.5)*t_sh + d_rad*t_ny))
            d_rl = (int(p_center[0] - (d_len*0.5)*t_ch - d_rad*t_nx), int(p_center[1] - (d_len*0.5)*t_sh - d_rad*t_ny))
            
            # 일체형 덕트 쉘
            pygame.draw.polygon(env.screen, (32, 38, 46), [d_fl, d_fr, d_rr, d_rl])
            pygame.draw.polygon(env.screen, (68, 78, 92), [d_fl, d_fr, d_rr, d_rl], 1)
            
            # 중앙 모터 코어 & 프로펠러
            m_f = (int(p_center[0] + 3*t_ch), int(p_center[1] + 3*t_sh))
            m_r = (int(p_center[0] - 4*t_ch), int(p_center[1] - 4*t_sh))
            pygame.draw.line(env.screen, (18, 22, 28), m_f, m_r, 3)
            
            prop_c = (int(p_center[0] - 1*t_ch), int(p_center[1] - 1*t_sh))
            p_b1 = (int(prop_c[0] - 3.5*t_nx), int(prop_c[1] - 3.5*t_ny))
            p_b2 = (int(prop_c[0] + 3.5*t_nx), int(prop_c[1] + 3.5*t_ny))
            pygame.draw.line(env.screen, (225, 235, 245), p_b1, p_b2, 2)

        # 중앙 회전식 라이다 센서 돔 (Rotating LiDAR Sensor Pod)
        Lidar_pos = TR((bx, by), -L*0.05, 0)
        pygame.draw.circle(env.screen, (32, 36, 42), Lidar_pos, 6)
        pygame.draw.circle(env.screen, (255, 215, 30), Lidar_pos, 3)
        # 라이다 360도 스캔 레이저 펄스 회전선
        scan_ang = env.frame * 0.35
        sp_x = int(Lidar_pos[0] + math.cos(scan_ang) * 6)
        sp_y = int(Lidar_pos[1] + math.sin(scan_ang) * 6)
        pygame.draw.line(env.screen, (0, 255, 200), Lidar_pos, (sp_x, sp_y), 2)
        pygame.draw.circle(env.screen, (0, 255, 200), (sp_x, sp_y), 2)

    def _draw_dashboard(self, hits):
        env = self.env
        bx, by = env.boat_pos
        h = env.boat_heading
        ch, sh = math.cos(h), math.sin(h)

        pygame.draw.rect(env.screen, (15, 35, 60), (0, env.sim_h, env.w, env.h - env.sim_h))
        pygame.draw.line(env.screen, (0, 180, 255), (0, env.sim_h), (env.w, env.sim_h), 3)

        # 마우스 커서 위치 확인 (호버 인터랙션)
        mpos = pygame.mouse.get_pos()

        # 체크박스 렌더링
        # 1. Show 1st Path (시안)
        cb1_row = getattr(env, 'cb1_row_rect', env.cb1_rect)
        cb1_hover = cb1_row.collidepoint(mpos)
        if cb1_hover:
            pygame.draw.rect(env.screen, (28, 56, 88), cb1_row, border_radius=4)
        pygame.draw.rect(env.screen, (255, 255, 255), env.cb1_rect, 2)
        if getattr(env, 'show_1st_path', True): pygame.draw.rect(env.screen, (0, 255, 200), env.cb1_rect.inflate(-6, -6))
        txt_col1 = (120, 255, 230) if cb1_hover else (255, 255, 255)
        env.screen.blit(self.font.render("Show 1st Path", True, txt_col1), (70, 670))

        # 2. Show 2nd Path (오렌지)
        cb2_row = getattr(env, 'cb2_row_rect', env.cb2_rect)
        cb2_hover = cb2_row.collidepoint(mpos)
        if cb2_hover:
            pygame.draw.rect(env.screen, (28, 56, 88), cb2_row, border_radius=4)
        pygame.draw.rect(env.screen, (255, 255, 255), env.cb2_rect, 2)
        if getattr(env, 'show_2nd_path', True): pygame.draw.rect(env.screen, (255, 140, 0), env.cb2_rect.inflate(-6, -6))
        txt_col2 = (255, 185, 95) if cb2_hover else (255, 255, 255)
        env.screen.blit(self.font.render("Show 2nd Path", True, txt_col2), (70, 706))

        # 3. Show Candidate WPs (연보라)
        cb3_row = getattr(env, 'cb3_row_rect', env.cb3_rect)
        cb3_hover = cb3_row.collidepoint(mpos)
        if cb3_hover:
            pygame.draw.rect(env.screen, (28, 56, 88), cb3_row, border_radius=4)
        pygame.draw.rect(env.screen, (255, 255, 255), env.cb3_rect, 2)
        if getattr(env, 'show_candidates', True): pygame.draw.rect(env.screen, (160, 180, 255), env.cb3_rect.inflate(-6, -6))
        txt_col3 = (195, 215, 255) if cb3_hover else (255, 255, 255)
        env.screen.blit(self.font.render("Show Candidate WPs", True, txt_col3), (70, 742))

        # 4. Show LiDAR Hits (소프트 옐로우)
        cb4_row = getattr(env, 'cb4_row_rect', env.cb4_rect)
        cb4_hover = cb4_row.collidepoint(mpos)
        if cb4_hover:
            pygame.draw.rect(env.screen, (28, 56, 88), cb4_row, border_radius=4)
        pygame.draw.rect(env.screen, (255, 255, 255), env.cb4_rect, 2)
        if env.show_lidar: pygame.draw.rect(env.screen, (225, 220, 130), env.cb4_rect.inflate(-6, -6))
        txt_col4 = (250, 245, 175) if cb4_hover else (255, 255, 255)
        env.screen.blit(self.font.render("Show LiDAR Hits", True, txt_col4), (70, 778))

        # 5. Show LiDAR Range (세이지 그린)
        cb5_row = getattr(env, 'cb5_row_rect', env.cb5_rect)
        cb5_hover = cb5_row.collidepoint(mpos)
        if cb5_hover:
            pygame.draw.rect(env.screen, (28, 56, 88), cb5_row, border_radius=4)
        pygame.draw.rect(env.screen, (255, 255, 255), env.cb5_rect, 2)
        if env.show_lidar_range: pygame.draw.rect(env.screen, (80, 175, 140), env.cb5_rect.inflate(-6, -6))
        txt_col5 = (130, 225, 180) if cb5_hover else (255, 255, 255)
        env.screen.blit(self.font.render("Show LiDAR Range", True, txt_col5), (70, 814))
        
        # 일시정지(PAUSE) 버튼
        is_paused = getattr(env, 'paused', False)
        p_bg = (240, 140, 30) if is_paused else (25, 45, 70)
        p_border = (255, 255, 255) if is_paused else (80, 120, 160)
        p_col = (10, 25, 45) if is_paused else (220, 230, 240)
        
        pygame.draw.rect(env.screen, p_bg, env.pause_btn, border_radius=4)
        pygame.draw.rect(env.screen, p_border, env.pause_btn, 2, border_radius=4)
        lbl_p = self.bold_font.render("||", True, p_col) if is_paused else self.font.render("||", True, p_col)
        env.screen.blit(lbl_p, (env.pause_btn.centerx - lbl_p.get_width()//2, env.pause_btn.centery - lbl_p.get_height()//2))

        # 배속 버튼
        cur_spd = getattr(env, 'sim_speed', 1)
        for spd, btn_rect in env.speed_btns.items():
            is_active = (cur_spd == spd and not is_paused)
            btn_bg = (0, 235, 255) if is_active else (25, 45, 70)
            btn_border = (255, 255, 255) if is_active else (70, 105, 145)
            border_w = 3 if is_active else 1
            text_color = (0, 15, 35) if is_active else (210, 225, 240)
            
            pygame.draw.rect(env.screen, btn_bg, btn_rect, border_radius=4)
            pygame.draw.rect(env.screen, btn_border, btn_rect, border_w, border_radius=4)
            
            lbl = self.bold_font.render(f"{spd}x", True, text_color) if is_active else self.font.render(f"{spd}x", True, text_color)
            env.screen.blit(lbl, (btn_rect.centerx - lbl.get_width()//2, btn_rect.centery - lbl.get_height()//2))

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

        # 180도 스캔 레이 라인 (저채도 세이지 그린)
        if env.show_lidar_range:
            for ang in env.rel_angles:
                if -math.pi/2 <= ang <= math.pi/2:
                    rx = pcx + math.sin(ang) * (env.lidar_range * scale_r)
                    ry = pcy - math.cos(ang) * (env.lidar_range * scale_r)
                    pygame.draw.line(self.pov_surf, (55, 120, 95), (pcx, pcy), (int(rx), int(ry)), 1)

        # 라이다 히트 포인트 렌더링 (저채도 소프트 옐로우)
        if env.show_lidar:
            for hp in hits:
                if hp is not None:
                    hdx = hp[0] - bx; hdy = hp[1] - by
                    hlf = hdx * f_vec[0] + hdy * f_vec[1]
                    hlr = hdx * r_vec[0] + hdy * r_vec[1]
                    if hlf >= -10:
                        pygame.draw.circle(self.pov_surf, (225, 220, 130), (int(pcx + hlr * scale_r), int(pcy - hlf * scale_r)), 2)

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
        show_1st = getattr(env, 'show_1st_path', getattr(env, 'show_paths', True))
        show_2nd = getattr(env, 'show_2nd_path', getattr(env, 'show_paths', True))
        if show_1st and env.current_wp is not None:
            dx_w = env.current_wp["pos"][0] - bx; dy_w = env.current_wp["pos"][1] - by
            lf_w = dx_w * f_vec[0] + dy_w * f_vec[1]; lr_w = dx_w * r_vec[0] + dy_w * r_vec[1]
            marker_objs.append(('wp1', lf_w, lr_w))

        if show_2nd and env.next_wp is not None:
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
                    pygame.draw.line(self.cam_surf, (0, 255, 220), (mx, 0), (mx, cam_h - 26), 2)
                    pygame.draw.circle(self.cam_surf, (0, 255, 220), (mx, 55), 6)
                    pygame.draw.circle(self.cam_surf, (255, 255, 255), (mx, 55), 2)
                    lbl_wp1 = self.micro_font.render("WP1", True, (0, 255, 220))
                    tx = mx + 8 if mx + 32 < cam_w else mx - lbl_wp1.get_width() - 8
                    self.cam_surf.blit(lbl_wp1, (tx, 49))
                elif obj_type == 'wp2':
                    pygame.draw.line(self.cam_surf, (200, 100, 255), (mx, 0), (mx, cam_h - 26), 2)
                    pygame.draw.circle(self.cam_surf, (200, 100, 255), (mx, 90), 6)
                    pygame.draw.circle(self.cam_surf, (255, 255, 255), (mx, 90), 2)
                    lbl_wp2 = self.micro_font.render("WP2", True, (200, 100, 255))
                    tx = mx + 8 if mx + 32 < cam_w else mx - lbl_wp2.get_width() - 8
                    self.cam_surf.blit(lbl_wp2, (tx, 84))
                elif obj_type == 'target':
                    pygame.draw.line(self.cam_surf, (20, 250, 80), (mx, 0), (mx, cam_h - 26), 3)
                    pygame.draw.circle(self.cam_surf, (20, 250, 80), (mx, 125), 7)
                    pygame.draw.circle(self.cam_surf, (255, 255, 255), (mx, 125), 3)
                    lbl_tgt = self.micro_font.render("Target", True, (20, 250, 80))
                    tx = mx + 9 if mx + 42 < cam_w else mx - lbl_tgt.get_width() - 8
                    self.cam_surf.blit(lbl_tgt, (tx, 119))

        # 패널 타이틀
        self.cam_surf.blit(self.font.render("LiDAR Gauge View", True, (255, 255, 255)), (10, 8))

        # 패널 하단 거리 색상 범례 도킹 바 (Legend HUD Bar)
        # 50px = 1m 기준: <70px (~1.4m), 70~140px (~2.8m), 140~220px (~4.4m), >220px (>4.4m)
        legend_bar_y = cam_h - 25
        pygame.draw.rect(self.cam_surf, (8, 16, 28, 235), (0, legend_bar_y, cam_w, 25))
        pygame.draw.line(self.cam_surf, (0, 140, 210), (0, legend_bar_y), (cam_w, legend_bar_y), 1)

        legend_items = [
            ((230, 60, 50), "<1.4m"),
            ((240, 160, 40), "<2.8m"),
            ((210, 210, 50), "<4.4m"),
            ((40, 170, 160), ">4.4m")
        ]
        
        # 총 4개 아이템을 패널 가로폭(320px)에 균등 정렬 배치 (small_font 18px로 시인성 향상)
        col_spacing = 76
        start_x = (cam_w - (col_spacing * 4 - 6)) // 2
        for i, (col, txt) in enumerate(legend_items):
            ix = start_x + i * col_spacing
            pygame.draw.rect(self.cam_surf, col, (ix, legend_bar_y + 7, 10, 10), border_radius=2)
            ltxt = self.small_font.render(txt, True, (225, 238, 255))
            self.cam_surf.blit(ltxt, (ix + 13, legend_bar_y + 5))

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
        show_1st = getattr(env, 'show_1st_path', getattr(env, 'show_paths', True))
        show_2nd = getattr(env, 'show_2nd_path', getattr(env, 'show_paths', True))
        if show_1st and env.current_wp is not None:
            dx_w = env.current_wp["pos"][0] - bx; dy_w = env.current_wp["pos"][1] - by
            lf_w = dx_w * f_vec[0] + dy_w * f_vec[1]; lr_w = dx_w * r_vec[0] + dy_w * r_vec[1]
            if lf_w > 2.0: overlay_objs.append(('wp1', lf_w, lr_w))

        if show_2nd and env.next_wp is not None:
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
                    lbl_wp1 = self.micro_font.render("WP1", True, (0, 255, 220))
                    tx = sx + 7 if sx + 30 < real_w else sx - lbl_wp1.get_width() - 7
                    self.real_cam_surf.blit(lbl_wp1, (tx, max(5, pole_y - 6)))
                elif obj_type == 'wp2':
                    pygame.draw.line(self.real_cam_surf, (200, 100, 255), (sx, sy_base), (sx, pole_y), 2)
                    pygame.draw.circle(self.real_cam_surf, (200, 100, 255), (sx, pole_y), 5)
                    pygame.draw.circle(self.real_cam_surf, (255, 255, 255), (sx, pole_y), 2)
                    lbl_wp2 = self.micro_font.render("WP2", True, (200, 100, 255))
                    tx = sx + 7 if sx + 30 < real_w else sx - lbl_wp2.get_width() - 7
                    self.real_cam_surf.blit(lbl_wp2, (tx, max(5, pole_y - 6)))
                elif obj_type == 'target':
                    pygame.draw.line(self.real_cam_surf, (20, 250, 80), (sx, sy_base), (sx, pole_y), 3)
                    pygame.draw.circle(self.real_cam_surf, (20, 250, 80), (sx, pole_y), 7)
                    pygame.draw.circle(self.real_cam_surf, (255, 255, 255), (sx, pole_y), 3)
                    lbl_tgt = self.micro_font.render("Target", True, (20, 250, 80))
                    tx = sx + 9 if sx + 40 < real_w else sx - lbl_tgt.get_width() - 9
                    self.real_cam_surf.blit(lbl_tgt, (tx, max(5, pole_y - 7)))

        pygame.draw.rect(self.real_cam_surf, (130, 180, 220), (0, 0, real_w, real_h), 2)
        self.real_cam_surf.blit(self.font.render("LiDAR 1st View", True, (255, 255, 255)), (10, 10))
        env.screen.blit(self.real_cam_surf, (1050, env.sim_h + 35))

        # --- 4. 실시간 베지어 곡선 & 곡률 프로파일 그래프 ---
        self._draw_bezier_profile()

        # --- 5. 웨이포인트 우선순위 가중치 비율 막대 게이지 ---
        self._draw_weight_breakdown()

    def _draw_bezier_profile(self):
        """우측 하단: 실시간 3차 베지어 곡선(Cubic S-Curve) 2D 궤적 그래프 (X: 전진거리, Y: 좌우편차 - 상하반전 및 3차 수식 표기)"""
        env = self.env
        bw, bh = 190, 220
        surf = self.bezier_surf
        surf.fill((10, 22, 38, 240))
        pygame.draw.rect(surf, (0, 180, 255), (0, 0, bw, bh), 2)
        
        # 타이틀
        surf.blit(self.bold_font.render("Bezier Curve 2D", True, (255, 255, 255)), (10, 8))
        
        # Y축 단위 표기
        surf.blit(self.small_font.render("Y(m)", True, (140, 190, 240)), (4, 30))
        
        # 그래프 영역
        gx, gy, gw, gh = 36, 42, 144, 94
        pygame.draw.rect(surf, (15, 30, 50), (gx, gy, gw, gh))
        pygame.draw.rect(surf, (40, 80, 120), (gx, gy, gw, gh), 1)
        
        y_center = gy + gh // 2
        
        path = getattr(env, 'bezier_path', None)
        bx, by = env.boat_pos
        h = env.boat_heading
        ch, sh = math.cos(h), math.sin(h)
        
        if not hasattr(self, 'scale_x_max'): self.scale_x_max = 4.0; self.scale_y_max = 2.0
        
        if path is not None and len(path) >= 4:
            pts = np.array(path)
            diffs = pts - env.boat_pos
            # 선박 기준 로컬 좌표계 변환 (X: 전방 거리, Y: 좌/우 편차 거리)
            x_loc = diffs[:, 0] * ch + diffs[:, 1] * sh
            y_loc = -diffs[:, 0] * sh + diffs[:, 1] * ch
            
            xm = x_loc / 50.0  # 미터 단위
            ym = y_loc / 50.0  # 미터 단위
            
            target_x_max = max(2.0, float(np.max(xm)) * 1.15)
            target_y_max = max(1.2, float(np.max(np.abs(ym))) * 1.35)
            
            self.scale_x_max = self.scale_x_max * 0.90 + target_x_max * 0.10
            self.scale_y_max = self.scale_y_max * 0.90 + target_y_max * 0.10
            
            sx_max = self.scale_x_max
            sy_max = self.scale_y_max
            
            # Y축 눈금선 및 수치 표기 (상단 +Y, 하단 -Y)
            surf.blit(self.small_font.render(f"+{sy_max:.1f}", True, (160, 200, 230)), (2, gy - 2))
            surf.blit(self.small_font.render(" 0.0", True, (0, 200, 255)), (2, y_center - 6))
            surf.blit(self.small_font.render(f"-{sy_max:.1f}", True, (160, 200, 230)), (4, gy + gh - 8))
            
            # 배경 중심 기준선 (Y = 0: 전방 직진선)
            pygame.draw.line(surf, (0, 140, 180), (gx, y_center), (gx + gw, y_center), 1)
            pygame.draw.line(surf, (25, 55, 80), (gx + gw//2, gy), (gx + gw//2, gy + gh), 1)
            
            # 3차 베지어 곡선 궤적 포인트 생성 (+ y_val 방향으로 위쪽 매핑)
            plot_pts = []
            for x_val, y_val in zip(xm, ym):
                px = int(gx + (x_val / max(0.1, sx_max)) * (gw - 12))
                py = int(y_center - (y_val / max(0.1, sy_max)) * (gh * 0.44))
                px = max(gx, min(gx + gw, px))
                py = max(gy, min(gy + gh, py))
                plot_pts.append((px, py))
                
            if len(plot_pts) >= 2:
                # 3차 함수 곡선 본체 (빛나는 시안색)
                pygame.draw.lines(surf, (50, 225, 255), False, plot_pts, 3)
                
            # 시작점 P0 (선박 원점)
            p0_x, p0_y = plot_pts[0]
            pygame.draw.circle(surf, (0, 255, 200), (p0_x, p0_y), 4)
            pygame.draw.circle(surf, (255, 255, 255), (p0_x, p0_y), 2)
            
            # 목표점 P3 (골/웨이포인트)
            p3_x, p3_y = plot_pts[-1]
            pygame.draw.circle(surf, (255, 160, 40), (p3_x, p3_y), 5)
            pygame.draw.circle(surf, (255, 255, 255), (p3_x, p3_y), 2)
            
            path_len_m = float(np.sum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))) / 50.0
            end_ym = ym[-1]
            
            # 실시간 3차 함수 계수 고속 계산 (y(0) = 0 제약 정규방정식 연산)
            if len(xm) >= 4 and (np.max(xm) - np.min(xm)) > 0.3:
                A = np.stack([xm**3, xm**2, xm], axis=1)
                ATA = A.T @ A
                ATA[0, 0] += 1e-4; ATA[1, 1] += 1e-4; ATA[2, 2] += 1e-4
                a, b, c = np.linalg.solve(ATA, A.T @ ym)
            else:
                a, b, c = 0.0, 0.0, 0.0
                
            if not hasattr(self, 'poly_coeffs'): self.poly_coeffs = (0.0, 0.0, 0.0)
            self.poly_coeffs = (
                self.poly_coeffs[0] * 0.80 + a * 0.20,
                self.poly_coeffs[1] * 0.80 + b * 0.20,
                self.poly_coeffs[2] * 0.80 + c * 0.20
            )
            sa, sb, sc = self.poly_coeffs
            formula_txt = f"y = {sa:+.3f}x³ {sb:+.2f}x² {sc:+.2f}x"
        else:
            sy_max = self.scale_y_max
            sx_max = self.scale_x_max
            surf.blit(self.small_font.render(f"+{sy_max:.1f}", True, (160, 200, 230)), (2, gy - 2))
            surf.blit(self.small_font.render(" 0.0", True, (0, 200, 255)), (2, y_center - 6))
            surf.blit(self.small_font.render(f"-{sy_max:.1f}", True, (160, 200, 230)), (4, gy + gh - 8))
            pygame.draw.line(surf, (0, 140, 180), (gx, y_center), (gx + gw, y_center), 1)
            pygame.draw.line(surf, (50, 225, 255), (gx, y_center), (gx + gw - 20, y_center), 3)
            path_len_m = 0.0
            end_ym = 0.0
            formula_txt = "y = +0.000x³ +0.00x² +0.00x"
            
        # X축 거리 눈금 및 수치 표기 (단위: m)
        surf.blit(self.small_font.render("0m", True, (140, 180, 220)), (gx, gy + gh + 2))
        mid_m_txt = f"{sx_max * 0.5:.1f}m"
        surf.blit(self.small_font.render(mid_m_txt, True, (140, 180, 220)), (gx + gw//2 - 10, gy + gh + 2))
        end_m_txt = f"{sx_max:.1f}m"
        surf.blit(self.small_font.render(end_m_txt, True, (140, 180, 220)), (gx + gw - len(end_m_txt)*7, gy + gh + 2))
        
        # 하단 실시간 3차 다항식 수식 표기 (실시간 수치 적용)
        surf.blit(self.small_font.render(formula_txt, True, (255, 220, 60)), (8, 164))
        
        # 하단 실시간 궤적 수치
        surf.blit(self.small_font.render(f"Len: {path_len_m:.1f}m | Lat Dev: {end_ym:+.1f}m", True, (220, 235, 255)), (8, 188))
        
        env.screen.blit(surf, (1385, env.sim_h + 35))

    def _draw_weight_breakdown(self):
        """우측 하단: 웨이포인트 우선순위 가중치 비율 분포 막대 게이지"""
        env = self.env
        ww, wh = 190, 220
        surf = self.weights_surf
        surf.fill((10, 22, 38, 240))
        pygame.draw.rect(surf, (0, 180, 255), (0, 0, ww, wh), 2)
        
        # 타이틀
        surf.blit(self.bold_font.render("WP Score Weights", True, (255, 255, 255)), (10, 10))
        
        wp = getattr(env, 'current_wp', None)
        factors = wp.get('factors', None) if wp is not None else None
        
        FACTOR_ITEMS = [
            ("Align", (0, 230, 255)),      # Heading Alignment
            ("Forward", (40, 225, 120)),   # Progress toward GPS
            ("Clear", (255, 215, 40)),     # Obstacle Clearance
            ("Perpend", (255, 140, 40)),   # Orthogonality to GPS (perp_exp)
            ("Proxim", (255, 90, 190)),    # Proximity to Boat
            ("Cluster", (170, 130, 255))   # Cluster Density
        ]
        
        bar_x = 64
        bar_w = 78
        bar_h = 8
        
        if wp is not None and factors is not None:
            # 가중치 w와 점수 raw를 곱하여 실제 기여 점수 산출 (w가 0이면 0% 표출)
            weighted_vals = []
            for k, _ in FACTOR_ITEMS:
                item = factors.get(k, factors.get('Perpend' if k == 'EffWidth' else None))
                if isinstance(item, dict):
                    w = float(item.get("w", 0.0))
                    raw = float(item.get("raw", 0.0))
                    weighted_vals.append(w * raw)
                elif isinstance(item, (int, float)):
                    weighted_vals.append(float(item))
                else:
                    weighted_vals.append(0.0)
                    
            tot = sum(weighted_vals)
            ratios = [v / tot if tot > 1e-6 else 0.0 for v in weighted_vals]
            
            for i, (name, col) in enumerate(FACTOR_ITEMS):
                y_pos = 46 + i * 27
                lbl = self.small_font.render(name, True, (210, 225, 240))
                surf.blit(lbl, (10, y_pos - 2))
                
                pygame.draw.rect(surf, (20, 40, 65), (bar_x, y_pos, bar_w, bar_h), border_radius=3)
                
                if ratios[i] > 0.001:
                    fill_w = max(2, int(ratios[i] * bar_w * 2.2))
                    fill_w = min(bar_w, fill_w)
                    pygame.draw.rect(surf, col, (bar_x, y_pos, fill_w, bar_h), border_radius=3)
                    
                pct = int(round(ratios[i] * 100))
                txt_pct = self.small_font.render(f"{pct}%", True, col if pct > 0 else (120, 140, 160))
                surf.blit(txt_pct, (bar_x + bar_w + 6, y_pos - 2))
        else:
            # 웨이포인트가 없을 때 (대기 상태 UI 유지)
            for i, (name, col) in enumerate(FACTOR_ITEMS):
                y_pos = 46 + i * 27
                lbl = self.small_font.render(name, True, (120, 145, 170))
                surf.blit(lbl, (10, y_pos - 2))
                
                pygame.draw.rect(surf, (20, 40, 65), (bar_x, y_pos, bar_w, bar_h), border_radius=3)
                
                txt_pct = self.small_font.render("--%", True, (90, 120, 150))
                surf.blit(txt_pct, (bar_x + bar_w + 6, y_pos - 2))
            
        env.screen.blit(surf, (1590, env.sim_h + 35))

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
            mode_txt = self.bold_font.render("AVOIDING", True, (255, 80, 60))
        elif has_wp:
            mode_txt = self.bold_font.render("GAP PASS", True, (0, 255, 220))
        else:
            mode_txt = self.bold_font.render("CRUISING", True, (50, 230, 120))
        hud_surf.blit(mode_txt, (10, 6))
        
        # 속도
        speed = float(np.linalg.norm(env.boat_vel))
        speed_knots = speed * 0.9
        spd_txt = self.small_font.render(f"Speed: {speed_knots:.1f} kt", True, (220, 235, 255))
        hud_surf.blit(spd_txt, (10, 32))
        
        # 속도 바 (실제 최고 속도 45.0 px/s 기준 정밀 스케일링)
        bar_w = 125
        pygame.draw.rect(hud_surf, (30, 50, 70), (10, 48, bar_w, 7))
        speed_ratio = float(np.clip(speed / 45.0, 0.0, 1.0))
        fill_w = int(speed_ratio * bar_w)
        bar_color = (255, 80, 60) if em else (0, 200, 100)
        pygame.draw.rect(hud_surf, bar_color, (10, 48, fill_w, 7))
        
        # 조타각
        steer_val = getattr(env, 'prev_steer', 0)
        steer_txt = self.small_font.render(f"Steer: {steer_val:+.2f}", True, (220, 235, 255))
        hud_surf.blit(steer_txt, (10, 60))
        
        # Heading (선박이 매 프레임 실시간 목표로 하는 각도, 라이다 좌표계: 북쪽 0도, 전방 90도, 남쪽 180도)
        tgt_h = getattr(env, 'heading_target', env.boat_heading)
        hdg_deg = (math.degrees(tgt_h) + 90) % 360
        hdg_txt = self.small_font.render(f"Heading: {hdg_deg:.0f}\u00b0", True, (220, 235, 255))
        hud_surf.blit(hdg_txt, (10, 76))
        
        # 목표 거리 (50px = 1m 기준 미터 단위 변환)
        d2t = float(np.linalg.norm(env.target - env.boat_pos))
        d2t_m = d2t / 50.0
        d2t_txt = self.small_font.render(f"Target: {d2t_m:.1f} m", True, (50, 230, 120))
        hud_surf.blit(d2t_txt, (10, 92))
        
        env.screen.blit(hud_surf, (hud_x, hud_y))