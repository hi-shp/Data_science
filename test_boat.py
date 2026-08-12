from stable_baselines3 import PPO
from boat_env import BoatEnv
import time

env = BoatEnv(render_mode=True)
model = PPO.load("boat_model_lidar")

obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    time.sleep(0.02)

env.close()
