import os
import torch
import numpy as np
import math

from train_behavior_cloning import DeepBoatAgent
from utils import wrap

_deep_agent = None
_device = None

def load_deep_agent(model_path="deep_agent_best.pth"):
    global _deep_agent, _device
    
    if _deep_agent is not None:
        return True
        
    if not os.path.exists(model_path):
        print(f"[Deep Nav] Error: Model weights not found at {model_path}")
        return False
        
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _deep_agent = DeepBoatAgent(input_dim=184, output_dim=2).to(_device)
    
    try:
        # Load weights
        _deep_agent.load_state_dict(torch.load(model_path, map_location=_device))
        _deep_agent.eval()
        print(f"[Deep Nav] Successfully loaded Behavior Cloning Model on {_device}")
        return True
    except Exception as e:
        print(f"[Deep Nav] Error loading model: {e}")
        return False

def get_deep_action(boat_pos, boat_heading, boat_vel, boat_ang_vel, target_pos, dists, lidar_range):
    """
    State를 구성하고 신경망을 통해 Motor Output(L, R)을 직접 추론합니다.
    """
    global _deep_agent, _device
    if _deep_agent is None:
        if not load_deep_agent():
            return 1500, 1500 # 기본값 (정지)
            
    # 1. State 추출 (generate_expert_data.py 와 완벽하게 동일해야 함)
    dist_to_target = np.linalg.norm(target_pos - boat_pos)
    target_angle = math.atan2(target_pos[1] - boat_pos[1], target_pos[0] - boat_pos[0])
    rel_target_angle = wrap(target_angle - boat_heading)
    
    lidar_norm = dists / lidar_range
    target_dist_norm = np.clip(dist_to_target / 500.0, 0.0, 1.0)
    target_angle_norm = rel_target_angle / np.pi
    
    vel_norm = np.linalg.norm(boat_vel)
    vel_norm_feat = np.clip(vel_norm / 100.0, 0.0, 1.0)
    ang_vel_norm = np.clip(boat_ang_vel / 5.0, -1.0, 1.0)
    
    state = np.concatenate([
        lidar_norm, 
        [target_dist_norm, target_angle_norm, vel_norm_feat, ang_vel_norm]
    ]).astype(np.float32)
    
    # 2. PyTorch 추론
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(_device)
    
    with torch.no_grad():
        out = _deep_agent(state_t).squeeze(0).cpu().numpy()
        
    # 3. Action 역정규화 ([-1, 1] -> [1100, 1900])
    out_L = out[0]
    out_R = out[1]
    
    # 역정규화 연산: out = (PWM - 1500) / 400
    # PWM = out * 400 + 1500
    pwm_L = out_L * 400.0 + 1500.0
    pwm_R = out_R * 400.0 + 1500.0
    
    # 클리핑 보호
    L = int(np.clip(pwm_L, 1100, 1900))
    R = int(np.clip(pwm_R, 1100, 1900))
    
    return L, R
