import pygame
import numpy as np
import math
import datetime
import os
from environment import BoatEnv
from perception import lidar_hits_np, update_grid, extract_clusters_from_grid, match_clusters
from deep_navigation import get_deep_action

def run():
    print("==================================================================")
    print("            Deep Learning Autonomous Navigation                   ")
    print("==================================================================")
    env = BoatEnv()

    while True:
        env.frame += 1
        
        env.update_dynamic_obstacles()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                return
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:
                    env.handle_click(e.pos)

        dists, hits = lidar_hits_np(
            env.boat_pos, env.boat_heading,
            env.rel_angles, env.dynamic_obstacles,
            env.lidar_range
        )

        update_grid(env.grid, hits)
        env.grid *= 0.945

        # 클러스터링 로직은 렌더링 목적과 상태 확인을 위해 유지
        new_c = extract_clusters_from_grid(env.grid)
        env.clusters, env.cluster_ids = match_clusters(
            env.clusters, env.cluster_ids, new_c
        )

        # ----------------------------------------------------
        # 인공신경망 추론 (복잡한 로직 전체를 1줄로 대체)
        # ----------------------------------------------------
        L, R = get_deep_action(
            boat_pos=env.boat_pos,
            boat_heading=env.boat_heading,
            boat_vel=env.boat_vel,
            boat_ang_vel=env.boat_ang_vel,
            target_pos=env.target,
            dists=dists,
            lidar_range=env.lidar_range
        )

        # UI에 표시하기 위해 기존 변수 초기화
        env.current_wp = None
        env.next_wp = None
        env.bezier_path = None
        env.pursuit_target = None
        env.next_bezier_path = None
        env.next_pursuit_target = None

        # ----------------------------------------------------
        env.step(L, R)

        env.render(hits)
        env.clock.tick(60)

        if env.collide() or np.linalg.norm(env.target - env.boat_pos) < 70:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            outdir = r"screenshot"
            if not os.path.exists(outdir):
                try:
                    os.makedirs(outdir)
                except:
                    pass
            p = os.path.join(outdir, f"{ts}.png")
            try:
                pygame.image.save(env.screen, p)
            except:
                pass
            env.reset()

if __name__ == "__main__":
    run()
