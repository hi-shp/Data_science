import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("boat_run_log.csv")
plt.plot(data["x"], data["y"], label="Path")
plt.scatter(data["x"].iloc[0], data["y"].iloc[0], c="green", label="Start")
plt.scatter(data["x"].iloc[-1], data["y"].iloc[-1], c="red", label="End")
plt.title("Boat Navigation Path (LiDAR-based)")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.axis("equal")
plt.legend()
plt.show()
