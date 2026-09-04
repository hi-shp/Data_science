#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_hyperparameters.py
밤샘 강화학습 / 진화 파라미터 최적화 탐색 스크립트

사용자 요구사항 반영:
1. 탐색 파라미터 범위:
   - align_exp: 5.0 ~ 10.0 (0.5 단위, 총 11개)
   - fwd_exp:   3.0 ~ 8.0  (0.5 단위, 총 11개)
   - clear_exp: 70 ~ 120   (5 단위, 총 11개)
2. 탐색 케이스:
   - Case 1 (1개씩 변경): align, fwd, clear 각각 1개씩 단독 변경 탐색
   - Case 2 (2개씩 변경): (align, fwd), (align, clear), (fwd, clear) 쌍 교차 변경 탐색
   - Case 3 (3개 다 변경): 3개 동시 탐색 및 엘리트 진화 강화학습(Evolutionary RL)을 통한 최고 성능 수렴
3. 각 케이스별 100번씩 4배속(sub_steps=4) 고속 시뮬레이션 평가
4. 최고 성공률 갱신 시 best_learned_params.json 및 리더보드 자동 갱신
5. 실시간 진행상황 training_log.txt 및 training_checkpoint.json 자동 저장 (중단 후 재개 가능)
"""

import os
import sys
import time
import json
import math
import argparse
import datetime
import itertools
from multiprocessing import Pool, cpu_count

# 헤드리스 고속 실행을 위한 더미 SDL 비디오 드라이버 설정
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
pygame.init()
import numpy as np

from environment import BoatEnv
from perception import lidar_hits_np, update_grid, extract_clusters_from_grid, match_clusters
from navigation import find_gap, target_is_clear, is_direct_target_safe, is_waypoint_switch_safe, is_front_blocked
from utils import wrap, make_bezier_path, pure_pursuit

CHECKPOINT_FILE = "training_checkpoint.json"
LEADERBOARD_FILE = "training_leaderboard.json"
LOG_FILE = "training_log.txt"
BEST_PARAMS_FILE = "best_learned_params.json"

BASE_PARAMS = {
    "steer_gain": 1.1,
    "steer_alpha": 0.3515,
    "mom_coeff": 0.00665,
    "pwm_rng": 270.36,
    "avoid_normal": 0.05,
    "avoid_em": 0.7,
    "clear_margin": 10,
    "em_enter": 125.0,
    "em_exit": 160.0,
    "em_hold_frames": 18,
    "align_exp": 8.0,
    "fwd_exp": 5.0,
    "clear_exp": 100.0,
    "width_exp": 0.2,
    "cluster_pen_w": 3.0,
    "wp_switch_thresh": 1.2,
    "perp_exp": 0.5,
    "prox_exp": 1.5
}

# 1개 케이스 평가 함수 (독립 워커 프로세스에서 실행)
def evaluate_case_worker(args):
    case_id, params_override, num_episodes = args
    
    # 기본 파라미터 복사 후 오버라이드
    params = dict(BASE_PARAMS)
    params.update(params_override)
    
    env = BoatEnv()
    env.sim_speed = 4
    env.params = params
    
    success_count = 0
    collision_count = 0
    timeout_count = 0
    total_frames = 0
    
    for ep in range(num_episodes):
        env.reset()
        ep_frames = 0
        
        while True:
            sub_steps = 4
            for _ in range(sub_steps):
                env.frame += 1
                ep_frames += 1
                env.update_dynamic_obstacles()

                dists, hits = lidar_hits_np(
                    env.boat_pos, env.boat_heading,
                    env.rel_angles, env.dynamic_obstacles,
                    env.lidar_range
                )

                update_grid(env.grid, hits)
                env.grid *= 0.945

                new_c = extract_clusters_from_grid(env.grid)
                env.clusters, env.cluster_ids = match_clusters(
                    env.clusters, env.cluster_ids, new_c
                )

                boat_spd = math.hypot(env.boat_vel[0], env.boat_vel[1])
                clear_to_target = is_direct_target_safe(
                    env.boat_pos, env.boat_heading, env.target,
                    env.dynamic_obstacles, env.boat_radius, boat_spd,
                    params=env.params
                )

                if clear_to_target:
                    new_wp = None
                    env.current_wp = None
                    env.next_wp = None
                    env.candidate_wps = []
                else:
                    new_wp = find_gap(
                        env.clusters, env.cluster_ids,
                        env.boat_pos, env.boat_heading,
                        env.target, env.visited,
                        env.grid, env.dynamic_obstacles,
                        params=env.params
                    )
                    if new_wp is not None:
                        env.candidate_wps = new_wp.get("candidates", [])
                    elif env.current_wp is not None:
                        env.candidate_wps = env.current_wp.get("candidates", [])
                    else:
                        env.candidate_wps = []

                if env.current_wp is not None:
                    vec_to_wp = env.current_wp["pos"] - env.boat_pos
                    dnow = np.linalg.norm(vec_to_wp)
                    wp_angle = math.atan2(vec_to_wp[1], vec_to_wp[0])
                    angle_diff = abs(wrap(wp_angle - env.boat_heading))
                    
                    if dnow < 25 or angle_diff > np.pi / 2 or target_is_clear(env.boat_pos, env.target, env.dynamic_obstacles):
                        p = env.current_wp["pair"]
                        env.visited.add(p)
                        env.visited.add((p[1], p[0]))
                        env.current_wp = None
                        env.candidate_wps = []

                if env.current_wp is not None:
                    id1, id2 = env.current_wp["pair"]
                    if id1 in env.cluster_ids and id2 in env.cluster_ids:
                        idx1 = env.cluster_ids.index(id1)
                        idx2 = env.cluster_ids.index(id2)
                        c1_now = env.clusters[idx1]
                        c2_now = env.clusters[idx2]
                        env.current_wp["c1"] = c1_now
                        env.current_wp["c2"] = c2_now
                        env.current_wp["pos"] = (c1_now + c2_now) / 2.0
                    elif new_wp is not None:
                        if is_waypoint_switch_safe(env.boat_pos, env.boat_heading, env.current_wp["pos"], new_wp["pos"], env.dynamic_obstacles, env.boat_radius, boat_spd, params=env.params):
                            env.current_wp = new_wp

                if new_wp is not None:
                    if env.current_wp is None:
                        env.current_wp = new_wp
                    elif new_wp["pair"] != env.current_wp["pair"] and new_wp["pair"] != (env.current_wp["pair"][1], env.current_wp["pair"][0]):
                        dist_to_curr = np.linalg.norm(env.current_wp["pos"] - env.boat_pos)
                        front_blocked = is_front_blocked(env.boat_pos, env.boat_heading, env.dynamic_obstacles, env.boat_radius, block_dist=120.0, fov_deg=65.0)
                        if not front_blocked and dist_to_curr > 80:
                            threshold = float(env.params.get('wp_switch_thresh', 1.1))
                            if new_wp["score"] > env.current_wp.get("score", 0.0) * threshold:
                                if is_waypoint_switch_safe(env.boat_pos, env.boat_heading, env.current_wp["pos"], new_wp["pos"], env.dynamic_obstacles, env.boat_radius, boat_spd, params=env.params):
                                    env.current_wp = new_wp

                if env.current_wp is not None and not clear_to_target:
                    temp_visited = env.visited.copy()
                    temp_visited.add(env.current_wp["pair"])
                    temp_visited.add((env.current_wp["pair"][1], env.current_wp["pair"][0]))
                    vec = env.current_wp["pos"] - env.boat_pos
                    next_head = math.atan2(vec[1], vec[0])
                    env.next_wp = find_gap(
                        env.clusters, env.cluster_ids,
                        env.current_wp["pos"], next_head,
                        env.target, temp_visited,
                        env.grid, env.dynamic_obstacles,
                        params=env.params
                    )
                else:
                    env.next_wp = None

                if env.current_wp is None:
                    env.bezier_path = make_bezier_path(env.boat_pos, env.boat_heading, env.target, obstacles=env.dynamic_obstacles, boat_radius=env.boat_radius, boat_speed=boat_spd)
                else:
                    env.bezier_path = make_bezier_path(env.boat_pos, env.boat_heading, env.current_wp["pos"], obstacles=env.dynamic_obstacles, boat_radius=env.boat_radius, boat_speed=boat_spd)
                
                if env.bezier_path is not None:
                    env.pursuit_target = pure_pursuit(env.bezier_path, env.boat_pos, lookahead=70)

                steer = env.update_steering(dists)
                if steer is None:
                    steer = 0

                L, R = env.get_pwm(steer)
                env.step(L, R)

                env.validate_wp_grid()
                env.validate_wp_obstacle_5x5()

                # 종료 판정
                is_collide = env.collide()
                dist_to_goal = np.linalg.norm(env.target - env.boat_pos)
                reached_goal = (dist_to_goal < 70)
                timed_out = (ep_frames > 2500)

                if is_collide or reached_goal or timed_out:
                    if reached_goal and not is_collide:
                        success_count += 1
                    elif is_collide:
                        collision_count += 1
                    else:
                        timeout_count += 1
                    total_frames += ep_frames
                    break

            if is_collide or reached_goal or timed_out:
                break

    success_rate = (success_count / num_episodes) * 100.0
    collision_rate = (collision_count / num_episodes) * 100.0
    avg_frames = total_frames / max(1, num_episodes)
    
    # 종합 피트니스: 성공률 최우선 + 충돌 페널티 + 빠른 주행 보너스
    fitness = success_rate * 10.0 - collision_rate * 5.0 - (avg_frames / 2500.0) * 2.0

    return {
        "case_id": case_id,
        "params": params_override,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "timeout_count": timeout_count,
        "avg_frames": avg_frames,
        "fitness": fitness
    }


class HyperparameterTrainer:
    def __init__(self, workers=6, episodes_per_case=100, resume=True):
        self.workers = workers
        self.episodes_per_case = episodes_per_case
        self.resume = resume
        
        # 탐색 격자 정의
        self.align_vals = [round(x, 1) for x in np.arange(5.0, 10.01, 0.5)]   # 11개
        self.fwd_vals = [round(x, 1) for x in np.arange(3.0, 8.01, 0.5)]       # 11개
        self.clear_vals = [int(x) for x in range(70, 121, 5)]                  # 11개
        
        self.history = []
        self.completed_cases = set()
        self.best_result = None
        self.start_time = time.time()
        
        self._load_checkpoint()

    def _log(self, msg):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
            f.flush()

    def _load_checkpoint(self):
        if self.resume and os.path.exists(CHECKPOINT_FILE):
            try:
                with open(CHECKPOINT_FILE, "r") as f:
                    data = json.load(f)
                    self.history = data.get("history", [])
                    self.completed_cases = set(d["case_id"] for d in self.history)
                    if self.history:
                        self.best_result = max(self.history, key=lambda x: (x["success_rate"], x["fitness"]))
                self._log(f"체크포인트 로드 완료: 기존 {len(self.completed_cases)}개 케이스 복구 완료.")
            except Exception as e:
                self._log(f"체크포인트 로드 오류: {e}")

    def _save_checkpoint(self):
        try:
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump({
                    "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_evaluated": len(self.history),
                    "best_result": self.best_result,
                    "history": self.history
                }, f, indent=2)
                
            # 리더보드 저장 (상위 20개)
            sorted_history = sorted(self.history, key=lambda x: (x["success_rate"], x["fitness"]), reverse=True)
            with open(LEADERBOARD_FILE, "w") as f:
                json.dump(sorted_history[:20], f, indent=2)
                
            # 역대 최고 성공률 갱신 시 best_learned_params.json 자동 업데이트
            if self.best_result:
                curr_best = dict(BASE_PARAMS)
                curr_best.update(self.best_result["params"])
                with open(BEST_PARAMS_FILE, "w") as f:
                    json.dump(curr_best, f, indent=2)
        except Exception as e:
            self._log(f"체크포인트 저장 오류: {e}")

    def run_batch(self, task_list, stage_name):
        # 이미 완료된 케이스 필터링
        pending_tasks = [(cid, p, self.episodes_per_case) for cid, p in task_list if cid not in self.completed_cases]
        total_tasks = len(task_list)
        already_done = total_tasks - len(pending_tasks)
        
        self._log(f"\n{'='*60}\n>> {stage_name} 시작 (총 {total_tasks}개 중 대기 {len(pending_tasks)}개 / 완료 {already_done}개)\n{'='*60}")
        
        if not pending_tasks:
            self._log(f"{stage_name}: 모든 케이스가 이미 완료되었습니다.")
            return

        with Pool(processes=self.workers) as pool:
            for res in pool.imap_unordered(evaluate_case_worker, pending_tasks):
                self.history.append(res)
                self.completed_cases.add(res["case_id"])
                
                # 최고 기록 확인
                is_new_best = False
                if self.best_result is None or res["success_rate"] > self.best_result["success_rate"]:
                    self.best_result = res
                    is_new_best = True
                elif res["success_rate"] == self.best_result["success_rate"] and res["fitness"] > self.best_result["fitness"]:
                    self.best_result = res
                    is_new_best = True
                    
                self._save_checkpoint()
                
                best_sr = self.best_result["success_rate"] if self.best_result else 0.0
                tag = "★ NEW BEST! ★" if is_new_best else ""
                self._log(f"[{len(self.completed_cases):4d}] {res['case_id']:<28} | 성공률: {res['success_rate']:5.1f}% | 충돌률: {res['collision_rate']:4.1f}% | (최고: {best_sr:5.1f}%) {tag}")

    def run(self):
        self._log("############################################################")
        self._log("   KABOAT 밤샘 하이퍼파라미터 탐색 및 강화학습 파이프라인 시작")
        self._log(f"   병렬 워커: {self.workers}개 | 케이스당 에피소드: {self.episodes_per_case}회 (4배속)")
        self._log("############################################################")

        # ---------------------------------------------------------
        # STAGE 1: 1개씩 단독 변경 탐색 (1D Sensitivity Sweep)
        # ---------------------------------------------------------
        stage1_tasks = []
        # 1-1. align_exp 변경
        for a in self.align_vals:
            stage1_tasks.append((f"S1_align_{a}", {"align_exp": a, "fwd_exp": 5.0, "clear_exp": 100.0}))
        # 1-2. fwd_exp 변경
        for f in self.fwd_vals:
            stage1_tasks.append((f"S1_fwd_{f}", {"align_exp": 8.0, "fwd_exp": f, "clear_exp": 100.0}))
        # 1-3. clear_exp 변경
        for c in self.clear_vals:
            stage1_tasks.append((f"S1_clear_{c}", {"align_exp": 8.0, "fwd_exp": 5.0, "clear_exp": c}))
            
        self.run_batch(stage1_tasks, "STAGE 1: 1개씩 단독 변경 탐색 (1D Sweep)")

        # Stage 1 분석: 각 파라미터별 상위 값 도출
        s1_results = [h for h in self.history if h["case_id"].startswith("S1_")]
        top_align = sorted([h for h in s1_results if "align" in h["case_id"]], key=lambda x: x["success_rate"], reverse=True)
        top_fwd = sorted([h for h in s1_results if "fwd" in h["case_id"]], key=lambda x: x["success_rate"], reverse=True)
        top_clear = sorted([h for h in s1_results if "clear" in h["case_id"]], key=lambda x: x["success_rate"], reverse=True)

        best_a = top_align[0]["params"]["align_exp"] if top_align else 8.0
        best_f = top_fwd[0]["params"]["fwd_exp"] if top_fwd else 5.0
        best_c = top_clear[0]["params"]["clear_exp"] if top_clear else 100.0
        self._log(f">> STAGE 1 완료! 개별 최적값: align={best_a}, fwd={best_f}, clear={best_c}")

        # ---------------------------------------------------------
        # STAGE 2: 2개씩 교차 변경 탐색 (2D Pairwise Sweep)
        # ---------------------------------------------------------
        stage2_tasks = []
        # 2-1. (align, fwd) 변경, clear 고정
        for a in self.align_vals:
            for f in self.fwd_vals:
                stage2_tasks.append((f"S2_AF_a{a}_f{f}", {"align_exp": a, "fwd_exp": f, "clear_exp": best_c}))
                
        # 2-2. (align, clear) 변경, fwd 고정
        for a in self.align_vals:
            for c in self.clear_vals:
                stage2_tasks.append((f"S2_AC_a{a}_c{c}", {"align_exp": a, "fwd_exp": best_f, "clear_exp": c}))
                
        # 2-3. (fwd, clear) 변경, align 고정
        for f in self.fwd_vals:
            for c in self.clear_vals:
                stage2_tasks.append((f"S2_FC_f{f}_c{c}", {"align_exp": best_a, "fwd_exp": f, "clear_exp": c}))
                
        self.run_batch(stage2_tasks, "STAGE 2: 2개씩 교차 변경 탐색 (2D Sweep)")

        # ---------------------------------------------------------
        # STAGE 3: 3개 다 변경 & 진화 강화학습 최적화 (3D Evolutionary RL Loop)
        # ---------------------------------------------------------
        self._log(f"\n{'='*60}\n>> STAGE 3: 3개 다 변경 및 엘리트 진화 강화학습 시작 (밤샘 연속 탐색)\n{'='*60}")
        
        # 상위 엘리트 후보 추출 (상위 8개)
        elite_candidates = sorted(self.history, key=lambda x: (x["success_rate"], x["fitness"]), reverse=True)[:8]
        population = [dict(c["params"]) for c in elite_candidates]
        
        generation = 1
        while True:
            self._log(f"\n--- [STAGE 3: 진화 세대 Gen {generation:03d}] (현재 최고 성공률: {self.best_result['success_rate']:.1f}%) ---")
            gen_tasks = []
            
            # 교차(Crossover) 및 변이(Mutation)를 통한 새로운 자손 생성
            for i in range(12):
                p1, p2 = np.random.choice(population, 2, replace=False)
                # 산술 교차
                alpha = np.random.uniform(0.3, 0.7)
                child_a = round(alpha * p1["align_exp"] + (1 - alpha) * p2["align_exp"], 1)
                child_f = round(alpha * p1["fwd_exp"] + (1 - alpha) * p2["fwd_exp"], 1)
                child_c = int(round(alpha * p1["clear_exp"] + (1 - alpha) * p2["clear_exp"]))
                
                # 변이 (Gaussian Mutation)
                if np.random.rand() < 0.6:
                    child_a = round(float(np.clip(child_a + np.random.choice([-1.0, -0.5, 0.5, 1.0]), 5.0, 10.0)), 1)
                if np.random.rand() < 0.6:
                    child_f = round(float(np.clip(child_f + np.random.choice([-1.0, -0.5, 0.5, 1.0]), 3.0, 8.0)), 1)
                if np.random.rand() < 0.6:
                    child_c = int(np.clip(child_c + np.random.choice([-15, -10, -5, 5, 10, 15]), 70, 120))
                    
                cid = f"S3_G{generation}_a{child_a}_f{child_f}_c{child_c}"
                gen_tasks.append((cid, {"align_exp": child_a, "fwd_exp": child_f, "clear_exp": child_c}))
                
            self.run_batch(gen_tasks, f"STAGE 3 Gen {generation}")
            
            # 차세대 개체군 업데이트 (전체 기록 중 상위 8개 유지)
            elite_candidates = sorted(self.history, key=lambda x: (x["success_rate"], x["fitness"]), reverse=True)[:8]
            population = [dict(c["params"]) for c in elite_candidates]
            generation += 1
            
            # 10세대마다 진행 리포트
            if generation % 10 == 0:
                elapsed_hours = (time.time() - self.start_time) / 3600.0
                self._log(f">> [중간 집계] 총 평가 케이스: {len(self.history)}개 | 경과 시간: {elapsed_hours:.2f}시간 | 최고 성공률: {self.best_result['success_rate']:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaboat Hyperparameter Overnight Trainer")
    parser.add_argument("--workers", type=int, default=min(8, max(1, cpu_count() - 2)), help="병렬 프로세스 개수 (기본값: 8)")
    parser.add_argument("--episodes", type=int, default=100, help="케이스당 평가 에피소드 수 (기본값: 100)")
    parser.add_argument("--no-resume", action="store_true", help="기존 체크포인트 무시하고 처음부터 시작")
    args = parser.parse_args()

    trainer = HyperparameterTrainer(
        workers=args.workers,
        episodes_per_case=args.episodes,
        resume=(not args.no_resume)
    )
    trainer.run()
