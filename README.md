# Run 3 — Deep Reinforcement Learning Agent

A PPO-based reinforcement learning agent that learns to play the browser game [Run 3](https://www.coolmathgames.com/0-run-3) directly from screen pixels, using no game API or internal state — only raw visual input captured in real time.

Built as a final project for CSCI 1470 (Deep Learning) at Brown University.

---

## Overview

This project trains a convolutional neural network to play Run 3, an endless-runner platformer where the player navigates a character across floating platforms in a rotating tunnel. The agent observes the game through screen capture, processes raw pixel data through a CNN backbone, and outputs actions (movement and jump commands) via simulated keyboard input.

The core challenge is learning a visuomotor policy from high-dimensional pixel observations with sparse, delayed reward signals — the agent must learn spatial awareness (platform detection, wall avoidance) and temporal reasoning (when to jump, how long to hold a direction) simultaneously.

### Key Technical Highlights

- **Screen-based learning** — the agent has zero access to game internals; it learns entirely from raw pixel observations captured via `mss`
- **Frame stacking** — 4 consecutive grayscale frames are stacked as input channels, giving the policy network temporal/motion information
- **Variable-duration actions** — the action space includes 3 hold durations (50ms, 100ms, 250ms) per direction, enabling fine-grained control
- **Multi-component reward shaping** — combines platform contact, progress tracking, wall proximity penalties, and survival bonuses
- **PPO with GAE** — Proximal Policy Optimization with Generalized Advantage Estimation for stable, sample-efficient policy updates
- **Special Game Tiles** — There are many subtly different types of tiles the agent can encounter–boost, falling, ramp. These are very difficult for the model to see and have drastically different effects. Our agent struggled with these.

---

## Architecture

### Neural Network (PPOActorCritic)

An actor-critic model with a shared CNN backbone (~2.2M parameters):

```
Input: (96, 96, 4) — 4 stacked grayscale frames, normalized to [0, 1]

┌─────────────────────────────────────────────┐
│  Conv2D  32 filters, 8×8, stride 4 → ReLU  │
│  Conv2D  64 filters, 4×4, stride 2 → ReLU  │
│  Conv2D  64 filters, 3×3, stride 1 → ReLU  │
│  Flatten → Dense 512 → ReLU                │
└──────────────┬──────────────────┬───────────┘
               │                  │
        ┌──────▼──────┐    ┌─────▼──────┐
        │ Actor Head  │    │ Critic Head│
        │ Dense 64    │    │ Dense 64   │
        │ Dense 10    │    │ Dense 1    │
        │ (logits)    │    │ (value)    │
        └─────────────┘    └────────────┘
```

**Actor** outputs logits over 10 discrete actions:

| Action | Key   | Hold Duration |
| ------ | ----- | ------------- |
| 0      | None  | —             |
| 1–3    | L/R/U | 50ms          |
| 4–6    | L/R/U | 250ms         |
| 7–8    | L+U/R+U | 50ms        |
| 9-10   | L+U/R+U | 250ms       |

**Critic** outputs a single scalar value estimate for the current state.

### Environment (Run3Env)

A custom environment that wraps the live game:

1. **Observation** — captures a 725×545 region of the screen, converts to 96×96 grayscale, and maintains a 4-frame stack
2. **Action execution** — simulates keyboard presses via PyAutoGUI with precise hold durations
3. **Game-over detection** — monitors a specific screen region for the white game-over dialog
4. **Reward computation** — evaluates platform contact and wall proximity from designated ROIs in the raw frame

### Reward Structure

| Component        | Value                    | Purpose                                     |
| ---------------- | ------------------------ | ------------------------------------------- |
| Alive bonus      | +0.1 per step            | Encourages survival                         |
| Platform contact | +3.0 × platform_ratio    | Rewards staying on the runway               |
| Progress         | +2.0 × max(Δplatform, 0) | Rewards increasing platform coverage        |
| Wall penalty     | −1.0 × wall_ratio        | Penalizes proximity to walls/side platforms |
| Time cost        | −0.005 per step          | Subtle pressure toward efficient movement   |
| Death penalty    | −20.0                    | Strong signal on game over                  |

Platform ratio and wall ratio are computed from pixel brightness in designated regions of interest (ROIs) anchored relative to the character's expected position.

---

## Training

### Algorithm: PPO with Clipped Surrogate Objective

Each training epoch:

1. **Rollout** — collect 1,024 environment steps using the current policy
2. **GAE** — compute Generalized Advantage Estimation (γ=0.99, λ=0.95)
3. **Update** — perform 3 passes of mini-batch SGD (batch size 64) on the clipped PPO objective

**Loss function:**

```
L = L_policy + 0.5 · L_value − 0.02 · H(π)
```

- `L_policy`: clipped surrogate objective (clip ratio ε=0.15)
- `L_value`: clipped value function MSE
- `H(π)`: entropy bonus for exploration

### Hyperparameters

| Parameter          | Value  |
| ------------------ | ------ |
| Learning rate      | 2×10⁻⁴ |
| Clip ratio (ε)     | 0.15   |
| Entropy coef       | 0.02   |
| Value coef         | 0.5    |
| Max gradient norm  | 0.5    |
| Steps per epoch    | 1,024  |
| Mini-batch size    | 64     |
| SGD epochs/rollout | 3      |
| Discount (γ)       | 0.99   |
| GAE lambda (λ)     | 0.95   |

### Experiment Tracking

Training metrics (returns, losses, entropy, evaluation scores) are logged to [Weights & Biases](https://wandb.ai/) for real-time monitoring and cross-run comparison.

---

## Project Structure

```
final-run3/
├── run3.ipynb          # All source code — environment, model, training loop
├── requirements.txt    # Python dependencies (pinned versions)
├── checkpoints/        # Saved model weights (.h5)
├── logs/               # Training logs
└── wandb/              # W&B experiment tracking data
```

---

## Setup

### Prerequisites

- Python 3.11
- macOS (screen capture coordinates are calibrated for Mac)
- Run 3 open in a browser, positioned to match the configured screen coordinates

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd final-run3

# Create and activate virtual environment
python3.11 -m venv run3_env
source run3_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Register Jupyter kernel
pip install jupyter notebook ipykernel
python -m ipykernel install --user --name=run3_env --display-name="Python 3.11 (Run3 Project)"
```

### Running

1. Open Run 3 in a browser and position the window so the game area aligns with the configured screen coordinates
2. Launch the notebook:
   ```bash
   jupyter notebook
   ```
3. Open `run3.ipynb` and select the **Python 3.11 (Run3 Project)** kernel
4. Run cells sequentially — the training loop will begin interacting with the live game

> **Note:** Screen coordinate constants (`TOP_X`, `TOP_Y`, etc.) are calibrated for a specific MacBook display. You may need to adjust these values for your setup. The debug visualization cell helps verify correct ROI placement.

---

## Tech Stack

| Category            | Tools                  |
| ------------------- | ---------------------- |
| Deep Learning       | TensorFlow 2.15, Keras |
| Computer Vision     | OpenCV 4.10            |
| Screen Capture      | MSS                    |
| Input Simulation    | PyAutoGUI              |
| Experiment Tracking | Weights & Biases       |
| Numerical           | NumPy 1.26             |
| Visualization       | Matplotlib             |
