import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 1. 신경망 아키텍처 정의 (MLP)
# ==========================================
class DeepBoatAgent(nn.Module):
    def __init__(self, input_dim=184, output_dim=2):
        super(DeepBoatAgent, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Mish(),
            
            nn.Linear(128, output_dim),
            nn.Tanh() # 출력을 [-1, 1]로 제한
        )
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. 메인 학습 스크립트
# ==========================================
def main():
    print("==================================================================")
    print("      딥러닝 행동 복제 (Behavior Cloning) 학습 스크립트        ")
    print("==================================================================")
    
    # 디바이스 설정 (GPU 지원)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] 현재 디바이스: {device}")
    
    # 1. 데이터 로드
    data_path = "data/expert_data.npz"
    if not os.path.exists(data_path):
        print(f"[Error] 데이터 파일이 존재하지 않습니다: {data_path}")
        return
        
    print("[Data] 전문가 데이터 로딩 중...")
    data = np.load(data_path)
    X = data['states']
    Y = data['actions']
    
    print(f"[Data] 입력(X) 크기: {X.shape}") # (N, 184)
    print(f"[Data] 출력(Y) 크기: {Y.shape}") # (N, 2)
    
    # 텐서 변환
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, Y_tensor)
    
    # 8:2 Train/Val Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    # 2. 모델 및 하이퍼파라미터 설정
    model = DeepBoatAgent().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    epochs = 20
    best_val_loss = float('inf')
    
    # --- 시각화(Matplotlib) 초기화 ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Behavior Cloning Training Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE Loss")
    ax.grid(True)
    line_train, = ax.plot([], [], label='Train Loss', color='blue', marker='o')
    line_val, = ax.plot([], [], label='Validation Loss', color='orange', marker='o')
    ax.legend()
    
    hist_epochs = []
    hist_train = []
    hist_val = []
    
    print("\n[Train] 모델 학습 시작...")
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        # TQDM Progress Bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]", leave=False)
        for batch_X, batch_Y in pbar:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            
            # 그래디언트 클리핑 (안정화)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            batch_loss = loss.item()
            train_loss += batch_loss * batch_X.size(0)
            pbar.set_postfix({'loss': f"{batch_loss:.6f}"})
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Val]", leave=False)
        with torch.no_grad():
            for batch_X, batch_Y in val_pbar:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                val_loss += loss.item() * batch_X.size(0)
                
        val_loss /= len(val_dataset)
        scheduler.step(val_loss)
        
        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {elapsed:.1f}s")
        
        # --- 시각화 그래프 업데이트 ---
        hist_epochs.append(epoch)
        hist_train.append(train_loss)
        hist_val.append(val_loss)
        
        line_train.set_data(hist_epochs, hist_train)
        line_val.set_data(hist_epochs, hist_val)
        
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "deep_agent_best.pth")
            print(f"  -> 새로운 최고 성능 모델 저장! (Val Loss: {val_loss:.6f})")

    print("\n[완료] 딥러닝 행동 복제 학습이 모두 종료되었습니다.")
    print("저장된 모델 가중치: 'deep_agent_best.pth'")
    
    plt.ioff()
    plt.savefig('training_loss.png')
    plt.show()

if __name__ == "__main__":
    main()
