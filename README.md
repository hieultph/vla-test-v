# G1 Humanoid Pick-and-Place via Fine-Tuned Vision-Language-Action Policy

Fine-tune **Isaac GR00T N1.7** on real Unitree G1 robot data to make the G1 respond to the command *"pick up all the items and put them in the yellow box"* in MuJoCo simulation, mapping camera images + language to direct joint position control.

---

## Demo

![Demo video placeholder](https://picsum.photos/seed/demo1/800/450)

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

The arm consistently moved toward the object. Grasp failed consistently. Root cause was later diagnosed as a data bug in the dataset conversion pipeline (see [Root Cause Analysis](#root-cause-analysis--data-bug)).

![Trial results placeholder](https://picsum.photos/seed/trials1/800/400)

---

## Code Structure

```
g1-challenge/
├── g1-manipulation-challenge/          # Simulation + inference
│   ├── run.py                          # Interactive keyboard control (explore the scene)
│   ├── vla_run.py                      # Closed-loop VLA inference — ZMQ client, obs saving
│   ├── teleoperate.py                  # Full joint teleoperation + episode recording (keyboard + gamepad)
│   ├── scene.xml                       # Modified scene: single table, red+green cubes, yellow box
│   ├── g1.xml                          # G1 robot MJCF (29 body DOF + 14 finger DOF, Dex3 hands)
│   ├── sim/tabletop_scene.xml          # Alternate tabletop scene
│   └── saved_obs/                      # Debug frames dumped during inference (.npz + .png)
│
└── Isaac-GR00T/                        # GR00T N1.7 — NVIDIA (submodule)
    ├── scripts/lerobot_conversion/     # Dataset format conversion (LeRobot v3 → v2)
    ├── scripts/finetune.sh             # Fine-tuning launcher (LoRA + W&B)
    ├── examples/G1_Dex3/              # G1 modality config (cameras, action/state mapping)
    └── gr00t/                          # Core model + ZMQ policy server
```

### Quick Start

```bash
# Clone with submodule
git clone --recurse-submodules <repo-url>
cd g1-manipulation-challenge
uv sync

# Explore the MuJoCo scene interactively (no GPU needed)
uv run python run.py

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

### Visual
- **Color domain shift**: The physical container in the real-robot dataset is **blue**. Applied HSV hue rotation to shift blue pixels to yellow across all video frames — bridging visual domain from real dataset to sim environment.

![HSV color shift before/after placeholder](https://picsum.photos/seed/hsv1/800/400)

- **Natural illumination variation**: Real-world capture inherently covers varying lighting conditions (no synthetic augmentation needed).

### Lingual
- **Language relabeling**: Replaced the generic annotation `"object_placement"` with `"pick up all the items and put them in the yellow box"` and paraphrases ("place all objects into the yellow container", "grab the items and drop them in the yellow box").

### Spatial
- Natural episode variation across 210 real-robot demonstrations covers diverse object placement positions — more effective than synthetic randomization on scripted demos.

---

## System Architecture

![System architecture diagram placeholder](https://picsum.photos/seed/arch1/800/500)

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Local Laptop (CPU)                           │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                      MuJoCo Simulation                        │    │
│  │                                                               │    │
│  │   G1 Robot (stationary)       4 Cameras                      │    │
│  │   28 DOF arm + hand   ──►    cam_left_high (480×640)         │    │
│  │   data.ctrl ◄── VLA           cam_right_high (480×640)       │    │
│  │   joint targets               cam_left_wrist                  │    │
│  │                               cam_right_wrist                 │    │
│  └──────────────────┬────────────────────┬────────────────────────┘   │
│              ZMQ REQ (obs)        ZMQ REP (actions)                  │
└─────────────────────┼────────────────────┼──────────────────────────┘
          CloudFlare Tunnel (zero-config, no VPN)
┌─────────────────────┼────────────────────┼──────────────────────────┐
│       RunPod: NVIDIA RTX PRO 5000 Blackwell (48 GB VRAM)             │
│  ┌──────────────────▼────────────────────▼────────────────────────┐  │
│  │                  GR00T N1.7 Policy Server                       │  │
│  │                                                                  │  │
│  │  Inputs:                            Outputs:                     │  │
│  │  • cam_left_high + cam_right_high   28D action chunk            │  │
│  │  • language prompt                  (16 steps ahead)            │  │
│  │  • 28D proprioception state                                      │  │
│  │                                                                  │  │
│  │  Backbone: DiT + flow-matching diffusion head                    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

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

Isaac Lab has a [beautiful built-in G1 pick-and-place environment](https://github.com/unitreerobotics/unitree_sim_isaaclab) — I would have used it if the hardware allowed. At 10 FPS the viewer was too laggy to interact with meaningfully. MuJoCo gave 40+ FPS on CPU.

![Isaac Lab vs MuJoCo placeholder](https://picsum.photos/seed/sim1/800/400)

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

![MuJoCo pose vs real dataset frame placeholder](https://picsum.photos/seed/pose1/800/400)

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

![W&B loss curve 2k steps placeholder](https://picsum.photos/seed/loss2k/800/400)

![W&B loss curve 6k steps placeholder](https://picsum.photos/seed/loss6k/800/400)

![W&B grad norm placeholder](https://picsum.photos/seed/grad1/800/400)

The 2k-step run showed clearly decreasing loss. Extended to 6k steps and monitored gradient norms to check for overfitting on the 210-episode dataset.

### Optimizer Note

Switched from AdamW to `adam_torch_fused` — PyTorch's fused CUDA implementation that uses Blackwell-specific kernel optimizations. Measured **~20% faster training throughput** on the RTX PRO 5000, reducing the 6k-step run by ~40 minutes with no change to hyperparameters or convergence.

---

## Evaluation Details

### Open-Loop Evaluation on Held-Out Test Set

![Open-loop eval: predicted vs ground-truth trajectory placeholder](https://picsum.photos/seed/eval1/800/400)

Ran open-loop evaluation on the 21 held-out test episodes. The model predicted reasonable arm trajectories but with a consistent timing bias — gripper closure was triggered earlier than it should be. This was the first signal pointing to the data bug below.

### Root Cause Analysis — Data Bug

During post-training debugging I found a **critical timestamp misalignment in the v3→v2 conversion script**.

Evidence found by comparing episode parquet and video files:

| Episode | Parquet action timestamps end at | Video duration |
|---------|:--------------------------------:|:--------------:|
| Episode 2 | 17.6 s | 22.0 s |
| Episode 3 | 17.6 s | 22.0 s |
| Consistent across multiple episodes | | |

![Parquet timestamp vs video duration placeholder](https://picsum.photos/seed/bug1/800/400)

The conversion script was truncating the action stream while the video stayed full-length. This caused **action labels to be offset relative to video frames** — the model learned to close the gripper before the hand had reached the object. This explains the "manipulation without seeing the object" bias seen in both open-loop evaluation and closed-loop trials.

This is a reproducible, well-scoped bug with a clear fix: align timestamps at conversion time, retrain. It is the primary reason grasp success is 0/5 in the current checkpoint.

---

## Key Engineering Challenges

### 1. No VR headset, no Isaac Lab — creative data sourcing

Without hardware for teleoperation or GPU headroom for Isaac Lab, I sourced the best available alternative: a real-robot G1 Dex3 dataset from HuggingFace. This required a full preprocessing pipeline — format conversion, HSV color shift, language relabeling, and camera alignment. The payoff is training on genuine dexterous manipulation data with natural motion variation, not scripted joint trajectories.

### 2. Cross-node inference without VPN

The GPU is on RunPod; the simulator runs locally. I needed a bridge without static IPs or VPN. Solution: GR00T's built-in ZMQ server + **CloudFlare Tunnel** (`cloudflared access tcp`). The tunnel exposes a public hostname routable from behind NAT. Latency was ~50–100 ms per request — acceptable since each call returns a 16-step action chunk that executes before the next query.

### 3. Finding the timestamp alignment bug

The model was closing the gripper before the hand reached the object. Initial hypotheses: action scaling, normalization drift, observation pipeline errors. I added `--save-obs-dir` to dump every inference request as images and confirmed the camera frames were correct. Then I moved upstream to inspect the dataset directly — where I found the video/parquet timestamp discrepancy. The discrepancy was consistent across episodes, pointing to a systematic conversion error rather than a one-off data issue.

### 4. Optimizer tuning on new GPU architecture

RTX PRO 5000 Blackwell is a recently released architecture. The default optimizer config doesn't fully leverage its kernel support. Switching to `adam_torch_fused` gave a measurable ~20% throughput improvement — a low-effort, high-return change for anyone training on Blackwell.

---

## What I'd Do Next

1. **Fix the timestamp bug** in `convert_v3_to_v2.py` and retrain — highest-leverage single change, expected to significantly improve grasp success
2. **Add wrist cameras** (`cam_left_wrist`, `cam_right_wrist`) to the modality config — the high cameras lack fine-grained spatial resolution needed for reliable finger placement around a small cube
3. **Collect targeted sim demos** using `teleoperate.py` (keyboard + gamepad) to supplement the dataset with the exact MuJoCo scene and task-specific language commands
4. **Tighten camera alignment** — match `head_cam` FOV and resolution more precisely to the real dataset's camera calibration to reduce visual distribution shift at inference
5. **Per-joint action normalization audit** — verify normalization statistics for the finger joints, which have much smaller ranges than arm joints and may be under-weighted in the diffusion loss

---

## References

1. **GR00T N1** — NVIDIA, *Isaac GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*, 2025. [[GitHub]](https://github.com/NVIDIA/Isaac-GR00T)
2. **Octo** — Octo Model Team, *Octo: An Open-Source Generalist Robot Policy*, arXiv:2405.12213, 2024.
3. **OpenVLA** — Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, arXiv:2406.09246, 2024.
4. **Diffusion Policy** — Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, arXiv:2303.04137, RSS 2023.
5. **RT-2** — Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, arXiv:2307.15818, CoRL 2023.
6. **Unitree G1 Dex3 Dataset** — Unitree Robotics, *G1 Dex3 Object Placement Dataset*, 2025. [[HuggingFace]](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ObjectPlacement_Dataset)
7. **Unitree XR Teleoperate** — Unitree Robotics, *XR Teleoperate for G1 data collection*, 2025. [[GitHub]](https://github.com/unitreerobotics/xr_teleoperate)
8. **Unitree Isaac Lab** — Unitree Robotics, *Unitree Sim IsaacLab*, 2025. [[GitHub]](https://github.com/unitreerobotics/unitree_sim_isaaclab)

---

*This was my first hands-on robotics project. I came in knowing none of these tools — MuJoCo physics, GR00T's architecture, LeRobot data format, ZMQ server protocols, CloudFlare tunneling, RTX Blackwell training — and built a working language-conditioned pick-and-place pipeline in 5 days. The timestamp bug I found late is the kind of subtle data quality issue that makes real robot learning genuinely hard. Diagnosing it clearly, tracing it to its source, and knowing the exact fix matters more to me than pretending the first training run worked perfectly.*
