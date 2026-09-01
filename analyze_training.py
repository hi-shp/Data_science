import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze():
    csv_file = "training_history.csv"
    if not os.path.exists(csv_file):
        print(f"File {csv_file} not found. Please run training first.")
        return

    df = pd.read_csv(csv_file)
    if len(df) < 2:
        print(f"Not enough data in {csv_file} (current rows: {len(df)}). Need at least 2 generations.")
        return

    print("==================================================================")
    print("       KABOAT Training Analysis & Parameter Causality Report      ")
    print("==================================================================")
    print(f"Total Generations Logged: {len(df)}")
    
    # 1단계 및 2단계 성공률 통계
    df["stage1_rate"] = pd.to_numeric(df["stage1_rate"], errors='coerce')
    df["stage1_cols"] = pd.to_numeric(df["stage1_cols"], errors='coerce')
    
    print(f"Success Rate - Min: {df['stage1_rate'].min():.1f}%, Max: {df['stage1_rate'].max():.1f}%, Mean: {df['stage1_rate'].mean():.1f}%")
    print(f"Collision Count - Min: {df['stage1_cols'].min()}, Max: {df['stage1_cols'].max()}, Mean: {df['stage1_cols'].mean():.1f}")
    print("------------------------------------------------------------------")
    print("  Parameter Correlation with Success Rate (Pearson & Spearman)")
    print("------------------------------------------------------------------")

    param_cols = [
        'steer_gain', 'steer_alpha', 'mom_coeff', 'pwm_rng',
        'avoid_normal', 'avoid_em', 'clear_margin',
        'em_enter', 'em_exit', 'em_hold_frames',
        'align_exp', 'fwd_exp', 'clear_exp', 'width_exp',
        'cluster_pen_w', 'wp_switch_thresh'
    ]

    corrs = []
    for col in param_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            p_corr = df[col].corr(df["stage1_rate"], method='pearson')
            s_corr = df[col].corr(df["stage1_rate"], method='spearman')
            if not np.isnan(p_corr):
                corrs.append((col, p_corr, s_corr))

    corrs.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"{'Parameter':20s} | {'Pearson r':>10s} | {'Spearman rho':>12s} | {'Impact'}")
    print("-" * 66)
    for name, p_c, s_c in corrs:
        impact = "POSITIVE (성공률 증가 기여)" if p_c > 0.15 else ("NEGATIVE (성공률 저하 유발)" if p_c < -0.15 else "NEUTRAL (미미한 영향)")
        print(f"{name:20s} | {p_c:10.4f} | {s_c:12.4f} | {impact}")

    # 시각화 그래프 생성
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # 1. 학습 곡선 (성공률 변화)
    ax1 = axes[0, 0]
    ax1.plot(df['generation'], df['stage1_rate'], 'b-o', label='Stage 1 (96 trials)', linewidth=2, markersize=4)
    if 'stage2_rate' in df.columns and df['stage2_rate'].notna().any():
        s2 = df[df['stage2_rate'].notna()]
        ax1.plot(s2['generation'], s2['stage2_rate'], 'r-s', label='Stage 2 (120 trials)', linewidth=2, markersize=5)
    ax1.set_title("Training Success Rate per Generation", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Success Rate (%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. 충돌 횟수 변화
    ax2 = axes[0, 1]
    ax2.plot(df['generation'], df['stage1_cols'], 'r-^', label='Collisions (out of 96)', linewidth=2, markersize=4)
    ax2.set_title("Collision Trend per Generation", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Collision Count")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. 파라미터별 상관계수 바 차트
    ax3 = axes[1, 0]
    names = [c[0] for c in corrs]
    values = [c[1] for c in corrs]
    colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in values]
    ax3.barh(names[::-1], values[::-1], color=colors[::-1])
    ax3.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax3.set_title("Parameter Correlation with Success Rate", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Pearson Correlation (r)")
    ax3.grid(True, alpha=0.3)

    # 4. 상위 주요 2개 파라미터 2D 산점도 / 군집 분석
    ax4 = axes[1, 1]
    if len(corrs) >= 2:
        top1, top2 = corrs[0][0], corrs[1][0]
        sc = ax4.scatter(df[top1], df[top2], c=df['stage1_rate'], cmap='viridis', s=60, edgecolors='black', alpha=0.8)
        cbar = plt.colorbar(sc, ax=ax4)
        cbar.set_label("Success Rate (%)")
        ax4.set_title(f"2D Parameter Cluster: {top1} vs {top2}", fontsize=12, fontweight='bold')
        ax4.set_xlabel(top1)
        ax4.set_ylabel(top2)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "Need more parameters logged", ha='center', va='center')

    save_path = "training_analysis.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n* Analysis visualization saved to: {save_path}")
    print("==================================================================")

if __name__ == "__main__":
    analyze()
