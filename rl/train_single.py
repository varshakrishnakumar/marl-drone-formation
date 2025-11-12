from stable_baselines3 import PPO
from sim.envs.single_drone_env import SingleDroneEnv
import matplotlib.pyplot as plt
import pandas as pd

# --- Initialize environment ---
env = SingleDroneEnv(gui=False)

# --- Train PPO model ---
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    tensorboard_log="./tb_logs/"
)

model.learn(total_timesteps=200_000)
model.save("ppo_single_hover")

# --- Evaluate trained policy ---
print("\n[INFO] Evaluating trained PPO policy...\n")
env.log_data = []  # clear any old training logs
obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, term, trunc, _ = env.step(action)
    if term or trunc:
        obs, _ = env.reset()

# --- Save evaluation log ---
env.save_log("hover_training_log.csv")
env.close()

# --- Plot rewards per episode ---
df = pd.read_csv("hover_training_log.csv")
df["episode"] = (df["step"] // 1000)

episode_means = df.groupby("episode")["reward"].mean()
plt.figure(figsize=(10, 5))
plt.plot(episode_means, label="Mean Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("PPO Hover Training Reward")
plt.legend()
plt.tight_layout()
plt.savefig("reward_curve.png")
plt.show()

# --- Plot mean altitude over episodes ---
plt.figure(figsize=(10, 5))
plt.plot(df.groupby("episode")["z"].mean(), label="Mean Height (m)", color="orange")
plt.xlabel("Episode")
plt.ylabel("Height (m)")
plt.title("Average Drone Altitude per Episode")
plt.legend()
plt.tight_layout()
plt.savefig("height_curve.png")
plt.show()

print("\n[INFO] Training complete. Logs and plots saved.\n")
