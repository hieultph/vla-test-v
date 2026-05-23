# G1 Humanoid Pick-and-Place via Fine-Tuned Vision-Language-Action Policy

Fine-tune **Isaac GR00T N1.7** on real Unitree G1 robot data to make the G1 respond to the command *"Pick up the red cube and place it in the yellow target region"* in MuJoCo simulation, mapping camera images + language to direct joint position control.

---

## Demo

<video src="./assets/videos/demo.mp4" controls width="100%"></video>

---

## Results

### 5-Trial Evaluation (Randomized Object Placement)

Command used in all trials: `"pick up all the items and put them in the yellow box"`

Object (cube) position randomized uniformly within the reachable right-side zone of the table each trial.

| Trial | Object position | Reach | Grasp | Lift | Transport | Place |
|-------|----------------|:-----:|:-----:|:----:|:---------:|:-----:|
| 1 | Random | ✅ | ❌ | ❌ | ❌ | ❌ |
| 2 | Random | ✅ | ❌ | ❌ | ❌ | ❌ |
| 3 | Random | ✅ | ⚠️ | ❌ | ❌ | ❌ |
| 4 | Random | ✅ | ❌ | ❌ | ❌ | ❌ |
| 5 | Random | ✅ | ⚠️ | ❌ | ❌ | ❌ |

**Reach: 5/5 — Full task: 0/5**

<!-- ![Trial results placeholder](https://picsum.photos/seed/trials1/800/400) -->

---

## Code Structure

```
g1-vin/
├── g1-pick-and-place/                                       # Simulation + inference
│   ├── ★ vla_run.py                                        # Closed-loop VLA inference — ZMQ client, obs saving
│   ├── ★ teleoperate.py                                    # Full joint teleoperation + episode recording (keyboard + gamepad)
│   ├── main.py                                              # Interactive MuJoCo viewer / manual joint control
│   ├── scene.xml                                            # Modified scene: single table, red cube, yellow box
│   └── g1.xml                                               # G1 robot MJCF (28 DOF: 7+7 arm + 7+7 Dex3 hand)
│
└── gr00t-unitree-g1/                                        # GR00T N1.7 — NVIDIA (submodule)
    ├── scripts/
    │   ├── ★ lerobot_conversion/convert_v3_to_v2.py        # Dataset format conversion (LeRobot v3 → v2)
    │   ├── ★ recolor_blue_to_yellow.py                     # HSV hue rotation — visual domain adaptation
    │   └── split_dataset.py                                 # Train / test split (90 / 10)
    ├── examples/
    │   ├── ★ finetune.sh                                   # LoRA fine-tuning launcher — wraps launch_finetune.py
    │   └── G1_Dex3/g1_dex3_config.py                       # G1 embodiment config (cameras, action/state mapping)
    └── gr00t/
        ├── ★ experiment/launch_finetune.py                 # Core fine-tuning entry point (invoked by finetune.sh)
        ├── ★ eval/run_gr00t_server.py                      # ZMQ policy server — inference endpoint (cloud GPU)
        ├── eval/open_loop_eval.py                           # Held-out test set evaluation (trajectory plots)
        └── policy/server_client.py                          # ZMQ client protocol
```

> **★ marks the files the challenge deliverables map to:**
> - `dataset/` → `lerobot_conversion/convert_v3_to_v2.py`, `recolor_blue_to_yellow.py`, `split_dataset.py`
> - `train/` → `finetune.sh` → `experiment/launch_finetune.py`, `G1_Dex3/g1_dex3_config.py`
> - `inference/` → `vla_run.py` (sim client), `run_gr00t_server.py` (policy server)

### Quick Start

```bash
# Clone with submodule
git clone --recurse-submodules <repo-url>
cd g1-pick-and-place
uv sync

# Explore the MuJoCo scene interactively (no GPU needed)
uv run python main.py

# Closed-loop VLA inference (requires GR00T server on cloud GPU)
cloudflared access tcp --hostname <your-tunnel> --url 127.0.0.1:5555 &
uv run python vla_run.py \
    --prompt "pick up all the items and put them in the yellow box" \
    --host 127.0.0.1 --port 5555

# Save every observation sent to the server for debugging
uv run python vla_run.py --prompt "..." --save-obs-dir ./saved_obs
```

---

## Action Tokenization: Handling G1's 28 DOF

The G1 with Dex3 hands has one of the most complex action spaces among humanoid robots.

### Joint Layout

```
Action vector (28D):
┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
│  left_arm[0:7]   │ right_arm[7:14]  │ left_hand[14:21] │right_hand[21:28] │
│                  │                  │                  │                  │
│  shoulder_pitch  │  shoulder_pitch  │  thumb_0         │  thumb_0         │
│  shoulder_roll   │  shoulder_roll   │  thumb_1         │  thumb_1         │
│  shoulder_yaw    │  shoulder_yaw    │  thumb_2         │  thumb_2         │
│  elbow           │  elbow           │  index_0         │  index_0         │
│  wrist_roll      │  wrist_roll      │  index_1         │  index_1         │
│  wrist_pitch     │  wrist_pitch     │  middle_0        │  middle_0        │
│  wrist_yaw       │  wrist_yaw       │  middle_1        │  middle_1        │
└──────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

GR00T's output is applied **directly to `data.ctrl`** as joint position targets — no intermediate policy layers or IK solver. The robot is stationary (pelvis fixed), so the full 28 DOF budget is available for bimanual arm + dexterous hand control.

### Continuous Diffusion Head vs. Discrete Tokens

GR00T N1.7 uses a **flow-matching diffusion head** that produces continuous joint position targets in an **action chunk of 16 timesteps**. This matters significantly for dexterous manipulation:

- **Discrete tokenization** (OpenVLA): each DOF bucketed into 256 bins. For a finger joint with ±1.5 rad range, that's ~12 ms precision per bin — often not enough to distinguish a firm grasp from a missed one.
- **Continuous diffusion** (GR00T): smooth trajectories, no quantization loss, temporal coherence across the 16-step chunk. The model produces gradual finger closure rather than sudden jumps — important for not knocking objects away on contact.

### Joint Positions vs. EEF Delta

| | Joint positions (used) | EEF delta + IK |
|--|--|--|
| Matches dataset format | ✅ | ❌ Requires re-labeling |
| No IK solver | ✅ | ❌ Adds failure modes |
| Works for 3-finger Dex3 hand | ✅ Direct | ❌ No clean EEF definition for 3-finger hand |
| Spatial generalization | ⚠️ Less intuitive in joint space | ✅ Cartesian more transferable |

For a 5-day sprint with a 28-DOF dexterous hand, joint positions were the practical choice. The dataset is labeled in joint space, GR00T's `REAL_G1` embodiment expects joint space, and there is no well-defined EEF for the Dex3 hand. Latent EEF representations are worth exploring as future work.

---

## Data Augmentation

### Techniques considered but not applied

A standard augmentation toolkit for this kind of real-to-sim VLA training would include:

| Category | Technique | Purpose |
|----------|-----------|---------|
| Visual | Random crop / resize | Robustness to camera FOV shift |
| Visual | Brightness / contrast jitter | Lighting invariance |
| Visual | Gaussian blur, noise injection | Sensor noise robustness |
| Visual | Random horizontal flip | Scene symmetry (with care for handedness) |
| Spatial | Object position jitter | Generalization beyond fixed placement |
| Spatial | Synthetic demo mixing | Coverage of edge-case grasp angles |
| Lingual | Paraphrase expansion (LLM-generated) | Instruction diversity |

These were intentionally skipped for two reasons:

1. **The real dataset is already high quality.** 210 episodes of genuine G1 Dex3 manipulation with natural lighting variation, real contact dynamics, and diverse object placement cover the training distribution well. Synthetic augmentation on top of real data risks introducing artifacts that hurt rather than help.

2. **Pipeline correctness first.** The priority for this sprint was to validate the full loop — data conversion → fine-tuning → ZMQ inference → sim execution — before tuning any knobs. Augmentation is an optimization step; applying it before the baseline works makes it impossible to attribute improvements or regressions to the right cause.

The only augmentation actually applied was the HSV hue rotation, which was a **necessity** (not an optimization) to bridge the real-dataset blue container to the sim's yellow target.

### Applied augmentation

#### Visual
- **Color domain shift**: The physical container in the real-robot dataset is **blue**. Applied HSV hue rotation to shift blue pixels to yellow across all video frames — bridging visual domain from real dataset to sim environment.

**Before/after HSV hue rotation:** Left frame shows the original blue container from the real-robot dataset; right frame shows the same frame after shifting blue pixels to yellow, matching the MuJoCo simulation target.

![HSV color shift before/after](./assets/images/ba-preprocess.png)

- **Natural illumination variation**: Real-world capture inherently covers varying lighting conditions (no synthetic augmentation needed).

#### Lingual
- **Language relabeling**: Replaced the generic annotation `"object_placement"` with `"pick up all the items and put them in the yellow box"` and paraphrases ("place all objects into the yellow container", "grab the items and drop them in the yellow box").

#### Spatial
- Natural episode variation across 210 real-robot demonstrations covers diverse object placement positions — more effective than synthetic randomization on scripted demos.

---

## System Architecture

**End-to-end inference pipeline:** Local MuJoCo simulation renders stereo camera frames, packages them with proprioception and language via ZMQ, and sends them through a CloudFlare tunnel to the GR00T N1.7 policy server running on a cloud GPU.

![System architecture diagram](./assets/images/system-architecture.png)


**Control loop**: Each iteration the client renders two camera frames, packages them with the current 28D joint state and language prompt, sends to the server, receives a 16-step action chunk, and writes joint positions directly to `data.ctrl`. The robot is fixed in place — no balance policy needed, the full action budget goes to arm and hand.

**Observation debugging**: Built `--save-obs-dir` that dumps every server request as `.npz` + `.json` + `.png`. This was critical for verifying what the model actually sees at inference time.

---

## Research: Model and Simulator Decisions

### Why GR00T N1.7 (not OpenVLA or Octo)

Before writing any training code I compared the three main open-source VLAs against this task's constraints:

| Criteria | **GR00T N1.7** | OpenVLA | Octo |
|----------|:-:|:-:|:-:|
| Native Unitree G1 embodiment | ✅ `UNITREE_G1` tag | ❌ | ❌ |
| Humanoid + bimanual design | ✅ | ⚠️ General | ⚠️ General |
| Data format | ✅ LeRobot v2 | ❌ RLDS | ⚠️ RLDS / JAX |
| Fine-tuning API | ✅ Single-command LoRA | ⚠️ Complex | ⚠️ Manual |
| Action prediction | ✅ Continuous diffusion | ⚠️ 256-bin discrete | ⚠️ Older arch |
| VRAM (inference) | ✅ ~16 GB+ | ❌ ~27 GB+ | ⚠️ Moderate |
| License | ✅ Apache 2.0 | ✅ MIT | ✅ MIT |

GR00T is the only model with native G1 embodiment, meaning the pretrained weight distribution already encodes G1 kinematics and scale — fine-tuning converges faster and requires fewer demonstrations than adapting a general-purpose backbone.

### Why MuJoCo (not Isaac Lab)

I ran both and measured:

| Simulator | FPS on my laptop | Decision |
|-----------|:----------------:|----------|
| **MuJoCo** | ~40 | Used — stable, headless-capable, full CPU physics |
| Isaac Lab | ~10 | Dropped — officially needs RTX 3080+ / 8 GB VRAM; unusable at 10 FPS |

Isaac Lab has a [beautiful built-in G1 pick-and-place environment](./assets/images/g1-isaac-sim.png) — I would have used it if the hardware allowed. At 10 FPS the viewer was too laggy to interact with meaningfully. MuJoCo gave 40+ FPS on CPU.

**Side-by-side simulator comparison:** Isaac Lab (left) renders the official Unitree G1 pick-and-place environment with rich visuals but ran at ~10 FPS on the local RTX 3050. MuJoCo (right) provides a lightweight CPU-only view that sustained ~40 FPS — the deciding factor for interactive development.

![Isaac Lab vs MuJoCo FPS comparison](./assets/images/isaac-sim-mujoco.png)

---

## Dataset

### Strategy: Real-World Data over Synthetic Demos

| Approach | Feasibility | Data quality | Decision |
|----------|:-:|:-:|----------|
| VR teleoperation ([Unitree XR](https://github.com/unitreerobotics/xr_teleoperate)) | ❌ No VR headset | High | Skipped |
| Scripted MuJoCo demos | ✅ Free | Low — scripted motion lacks dexterity variation | Skipped |
| **Real Unitree G1 Dex3 dataset (HuggingFace)** | ✅ Free download | High — real robot, natural variation | **Used** |

Training on real robot data eliminates the sim-to-real gap in the training signal. GR00T's backbone already knows G1 kinematics — fine-tuning on real data teaches the manipulation task without fighting distribution shift in the dynamics.

### Dataset: [Unitree G1 Dex3 ObjectPlacement](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ObjectPlacement_Dataset)

| Property | Value |
|----------|-------|
| Episodes | 210 |
| Total frames | 98,266 |
| Frame rate | 30 fps |
| Action / state space | 28D joint positions |
| Cameras | `cam_left_high`, `cam_right_high`, `cam_left_wrist`, `cam_right_wrist` |
| Original language annotation | `"object_placement"` |
| Train / test split | 189 / 21 episodes (90 / 10) |

### Preprocessing Pipeline

**1. Format conversion** — Dataset is LeRobot v3; GR00T N1.7 requires LeRobot v2. Used `Isaac-GR00T/scripts/lerobot_conversion/convert_v3_to_v2.py`.

**2. Visual domain adaptation** — HSV hue rotation to shift the blue container to yellow, matching the MuJoCo scene.

**3. Language relabeling** — Replaced `"object_placement"` with `"pick up all the items and put them in the yellow box"`.

**4. Sim-to-dataset alignment** — Extracted the initial joint configuration from a real episode, injected it into MuJoCo, then manually tuned the `head_cam` FOV and position to match the dataset's first training frame. This ensures what the model sees at inference closely matches the training distribution.

**Visual domain alignment:** Left is a frame from the real-robot dataset; right is the MuJoCo scene after manually tuning head-cam FOV, position, and object placement to match the dataset's first training frame.

![MuJoCo pose vs real dataset frame](./assets/images/setting-like-real.png)

**5. Pose validation** — Verified all 28 DOF values from the real robot dataset were within the joint limits defined in `g1.xml`.

---

## Training

### Configuration

| Parameter | Value |
|-----------|-------|
| Base model | GR00T N1.7 — `REAL_G1` embodiment |
| Fine-tuning method | LoRA |
| Cameras used | `cam_left_high`, `cam_right_high` |
| Max steps | 2,000 → extended to 6,000 |
| Global batch size | 64 |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Episode sampling rate | 0.5 |
| Action horizon | 16 steps |
| Dataloader workers | 8 |

### Infrastructure

| Task | Hardware | Cost |
|------|----------|------|
| Data preprocessing, conversion, scene alignment | Local laptop — RTX 3050 6 GB, 16 GB RAM, Ubuntu 22.04 | Free |
| Fine-tuning + inference | RunPod — NVIDIA RTX PRO 5000 Blackwell, 48 GB VRAM | ~$8–12 |

### Training Curves (W&B)

**Training loss (first 2,000 steps):** Diffusion loss and flow-matching loss both decrease monotonically, indicating the LoRA adapter is learning the task distribution without divergence.

![W&B loss curve 2k steps](./assets/images/train-loss.jpg)

**Training loss (extended to 6,000 steps):** Loss continues to trend downward. The extended run confirms the earlier trend is stable rather than lucky early convergence.

![W&B loss curve 6k steps](./assets/images/train-loss-6k.jpg)

**Gradient norm:** Gradient norms stay within a healthy range (~0.1–0.5) throughout training.

![W&B grad norm](./assets/images/train-grad_norm.jpg)

The 2k-step run showed clearly decreasing loss. Extended to 6k steps and monitored gradient norms to check for overfitting on the 210-episode dataset.

### Optimizer Note

Switched from AdamW to `adam_torch_fused` — PyTorch's fused CUDA implementation that uses Blackwell-specific kernel optimizations. Measured **~15% faster training throughput** on the RTX PRO 5000.

![Adam_torch_fused](./assets/images/adamw-torch-fused.png)

---

## Evaluation Details

### Open-Loop Evaluation on Held-Out Test Set

Training is **okay but not good**. The model has learned the rough structure of the task — arm motion follows the right direction and general phasing — but the predicted trajectories (orange) are noticeably unstable compared to ground truth (blue): the lines oscillate, drift, and deviate rather than tracking smoothly. The model has not converged to precise, stable imitation.

**Test Sample 0 — Episode 180:** Predicted (orange) vs. ground-truth (blue) joint trajectories. Arm motion tracks well; finger closure begins ~0.3 s earlier than labeled.

![Open-loop eval sample 0](./assets/images/traj_0.jpeg)

**Test Sample 1 — Episode 181:** Similar early-gripper pattern. The model closes thumb and index joints before the hand has reached the object height.

![Open-loop eval sample 1](./assets/images/traj_1.jpeg)

**Test Sample 2 — Episode 182:** Wrist yaw drift visible in the second half. Despite drift, the temporal offset of the grasp phase is consistent with Samples 0–1.

![Open-loop eval sample 2](./assets/images/traj_2.jpeg)

**Test Sample 3 — Episode 183:** Larger positional deviation in shoulder roll, but the overall action chunk shape is preserved. Grasp timing bias remains the dominant error mode.

![Open-loop eval sample 3](./assets/images/traj_3.jpeg)

**Test Sample 4 — Episode 184:** Best-tracking sample of the five. Even here, the finger joints cross the closure threshold ~2–3 frames ahead of the label.

![Open-loop eval sample 4](./assets/images/traj_4.jpeg)

> **Pattern summary across 21 held-out episodes:** The model consistently predicts reasonable arm trajectories, but gripper closure is systematically triggered earlier than the ground truth. This timing bias was the first signal pointing to the data bug below.

### Root Cause Analysis — Data Bug

> **Status: Fixed.** The timestamp misalignment described below has been patched in the conversion script. This section documents the bug for reference — if you reproduce training from scratch and see early-gripper bias, this is the likely cause.

During post-training debugging I found a **critical timestamp misalignment in the v3→v2 conversion script**.

Evidence found by comparing episode parquet and video files:

| Episode | Parquet action timestamps end at | Video duration |
|---------|:--------------------------------:|:--------------:|
| Episode 2 | 17.6 s | 22.0 s |
| Episode 3 | 17.6 s | 22.0 s |
| Consistent across multiple episodes | | |

**Timestamp mismatch evidence:** Parquet action rows end at 17.6 s while the corresponding video file runs to 22.0 s, leaving ~4.4 s of unlabeled frames at the end of every converted episode.

![Parquet timestamp vs video duration](./assets/images/parquet-vs-video-timestamp.png)

The conversion script was truncating the action stream while the video stayed full-length. This caused **action labels to be offset relative to video frames** — the model learned to close the gripper before the hand had reached the object. This explains the "manipulation without seeing the object" bias seen in both open-loop evaluation and closed-loop trials.

The fix is to align timestamps at conversion time so action rows and video frames stay in sync. If you re-run the conversion pipeline without the patch and observe systematic early-gripper behavior in open-loop evaluation, check the parquet/video timestamp discrepancy first.

---

## What I'd Do Next

1. **Add wrist cameras** (`cam_left_wrist`, `cam_right_wrist`) to the modality config — the high cameras lack fine-grained spatial resolution needed for reliable finger placement around a small cube
2. **Collect targeted sim demos** using `teleoperate.py` (keyboard + gamepad) to supplement the dataset with the exact MuJoCo scene and task-specific language commands
3. **Tighten camera alignment** — match `head_cam` FOV and resolution more precisely to the real dataset's camera calibration to reduce visual distribution shift at inference
4. **Per-joint action normalization audit** — verify normalization statistics for the finger joints, which have much smaller ranges than arm joints and may be under-weighted in the diffusion loss

---

## References

**Models**

1. **GR00T N1** — NVIDIA, *Isaac GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*, 2025. [[GitHub]](https://github.com/NVIDIA/Isaac-GR00T)
2. **Octo** — Octo Model Team, *Octo: An Open-Source Generalist Robot Policy*, arXiv:2405.12213, 2024. [[Website]](https://octo-models.github.io/) [[GitHub / Fine-tune]](https://github.com/octo-models/octo)
3. **OpenVLA** — Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, arXiv:2406.09246, 2024. [[GitHub]](https://github.com/openvla/openvla)
4. **OpenVLA-OFT** — *OpenVLA-OFT: Efficient Fine-Tuning of OpenVLA*, 2025. [[Website]](https://openvla-oft.github.io/) [[arXiv 1]](https://arxiv.org/html/2512.11921v1) [[arXiv 2]](https://arxiv.org/html/2603.16044v1)
5. **Diffusion Policy** — Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, arXiv:2303.04137, RSS 2023.
6. **RT-2** — Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, arXiv:2307.15818, CoRL 2023.
7. **Psi0** — Physical Superintelligence Lab, *Psi0: Open-Source Robot Policy*, 2025. [[GitHub]](https://github.com/physical-superintelligence-lab/Psi0)

**Unitree G1 Hardware & Simulation**

8. **Unitree G1 Developer Guide** — Unitree Robotics, *G1 Developer Documentation*, 2025. [[Docs]](https://support.unitree.com/home/zh/G1_developer/about_G1) [[Remote Control]](https://marketing.unitree.com/article/zh/G1/Remote_Control.html)
9. **Unitree G1 Joint Mapping (CSDN)** — Community reference for G1 joint layout and control interface, 2025. [[Article]](https://blog.csdn.net/qq_28912651/article/details/149045107)
10. **Unitree XR Teleoperate** — Unitree Robotics, *XR Teleoperate for G1 data collection*, 2025. [[GitHub]](https://github.com/unitreerobotics/xr_teleoperate)
11. **Unitree Isaac Lab** — Unitree Robotics, *Unitree Sim IsaacLab*, 2025. [[GitHub]](https://github.com/unitreerobotics/unitree_sim_isaaclab)

**Dataset & Data Pipeline**

12. **Unitree G1 Dex3 Dataset** — Unitree Robotics, *G1 Dex3 Object Placement Dataset*, 2025. [[HuggingFace]](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ObjectPlacement_Dataset)
13. **RLDS Dataset Builder** — Pertsch et al., *RLDS Dataset Builder — converting datasets to RLDS format*, 2023. [[GitHub]](https://github.com/kpertsch/rlds_dataset_builder)

**Community & Blogs**

14. **Open-Weight Robot Models Overview** — RoboCloud, 2025. [[Blog]](https://robocloud-dashboard.vercel.app/learn/blog/open-weight-robot-models)
15. **G1 Pick-and-Place Implementation Notes (CSDN)** — Community walkthrough of G1 manipulation setup, 2024. [[Article]](https://blog.csdn.net/qq_41204464/article/details/159830799)

---
