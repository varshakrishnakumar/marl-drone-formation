# Multi-Agent Reinforcement Learning for Drone Formation Control

This repository contains simulation and reinforcement learning code for **ASTE 599: Extreme Environment Robotic Autonomy (Fall 2025)**.  
We train multi-agent controllers for drone formation flight and obstacle avoidance using **PyBullet** and **Reinforcement Learning (PPO/SAC)** frameworks.

---

## Overview
The project develops a MARL (Multi-Agent Reinforcement Learning) system that enables multiple quadrotors to:
- Maintain formation relative to a leader or virtual center.
- Avoid inter-agent and environmental collisions.
- Adapt in real time to dynamic obstacles.

The system is trained in simulation (PyBullet) and benchmarked against classical control baselines (PID / LQR).

---

## Repository Structure
| Folder | Description |
|---------|-------------|
| `sim/` | PyBullet environments and quadrotor physics models. |
| `rl/` | RL algorithms (PPO/SAC), training scripts, reward shaping. |
| `systems/` | Logging, visualization tools, and ROS2 bridge stubs. |
| `notebooks/` | Analysis notebooks and experiment results. |
| `data/` | Training logs, checkpoints, and evaluation outputs. |
| `docs/` | Design documents, plots, and summaries. |

---

## Environment Setup

#### macOS / Linux / Windows
```bash
# 1. Create environment
conda create -n marl python=3.10

# 2. Activate environment
conda activate marl

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) Verify setup
python -m torch.utils.collect_env
python -c "import pybullet; print('PyBullet OK')"
python -c "import stable_baselines3; print('SB3 OK')"
