from stable_baselines3 import PPO
from boat_env import BoatEnv

env = BoatEnv(render_mode=False)
model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=5000)  # 충분히 학습시키기
model.save("boat_model_lidar")  # 모델 저장

env.close()
