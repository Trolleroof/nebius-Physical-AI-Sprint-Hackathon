# Physical AI Sprint — Continual Embodied Learning
**Hackathon:** The Physical AI Sprint alongside Actuate SF  
**Hosts:** Nebius · NVIDIA · Antioch · Toloka  
**Date:** August 17, 2026  
**Robot:** SO-101  
**Project direction:** Sim → Real → Diagnose → Adapt → Learn → Redeploy

---

# 0. Executive Summary

## One-line pitch

**A robot that does not stop learning when it gets deployed.**

## Technical one-line pitch

**A real-to-sim-to-real data flywheel for continual robot policy improvement, using an embodied reasoning critic to turn real-world failures into targeted simulation curriculum.**

## Core idea

Most robot-learning systems look like:

```text
collect data → train → evaluate → deploy → stop
```

We want:

```text
collect → train → verify → deploy
                         ↓
                     experience
                         ↓
                      diagnose
                         ↓
                  adapt simulation
                         ↓
                    collect more
                         ↓
                      retrain
                         ↓
                     redeploy
                         ↺
```

The policy first learns in simulation. Once it passes simulation evaluation, it is deployed to the real SO-101. If reality exposes a failure, an embodied reasoning critic watches the real rollout, identifies the failure mode, and maps that diagnosis into a new simulation distribution. The simulator then generates targeted corrective experience, the policy is retrained, and the improved policy is redeployed.

The important story is not simply “we trained ACT.”

The story is:

> **Deployment becomes the next training phase.**

---

# 1. Official Hackathon Constraints

## Challenge

Physical AI requires closing the loop across:

- **Perception:** understand camera and sensor data
- **Reasoning:** turn goals into multi-step behavior and adapt to change
- **Action:** turn plans into robot motion that works

Our project deliberately covers all three.

## Chosen project direction

The official packet allows:

1. Simulation only
2. Hardware only
3. **Sim + real**

We are choosing **sim + real**.

## Hardware

Primary robot:

- SO-101 leader/follower arm pair

Primary task:

- Green cube → blue tray

Fallback:

- Reach-to-pose / recovery task if props are unavailable

## Platform

Antioch provides cloud robotics simulation on Nebius GPUs using the NVIDIA Isaac stack.

We do **not** need to install Isaac locally.

Antioch gives us:

- Isaac Sim / Isaac Lab execution in the cloud
- browser streaming
- parameterized scenarios
- headless batch execution
- metrics and logs
- replayable runs
- shared team development environment

## Judging

The four equally weighted categories are:

1. **Ambition**
2. **Functionality**
3. **Creativity**
4. **Architectural quality**

## Critical ground rule

Do not spend the day trying to fine-tune a giant VLA.

The official guide explicitly supports training a small ACT or Diffusion policy in the available time.

That is why the learned control policy is intentionally small.

---

# 2. Project Thesis

## Narrative framing

The best high-level framing is:

> **Robots should learn on the job.**

More precise terminology:

- **Continual embodied learning**
- **Continual policy improvement**
- **Deployment-driven continual adaptation**
- **Real-to-sim-to-real learning loop**
- **Robot data flywheel**

For this specific implementation, the most technically accurate phrase is:

> **Deployment-driven continual policy improvement through critic-guided, failure-conditioned data aggregation.**

## Is this “off-policy learning”?

Not really.

“On-policy” and “off-policy” are mainly reinforcement-learning distinctions.

Our core ACT loop is imitation learning:

```text
deployment failure
      ↓
critic identifies weak condition
      ↓
simulator generates targeted corrective demonstrations
      ↓
append new demonstrations to dataset
      ↓
retrain ACT
```

If we later add RL updates using rewards collected under older policies, then “off-policy RL” could become relevant.

For the hackathon, do not pitch the system as off-policy learning.

## Is this strict continual learning?

“Continual learning” is a good narrative description, but strict continual-learning research often also evaluates catastrophic forgetting across sequential tasks.

We are more specifically demonstrating:

**continual adaptation / continual policy improvement after deployment**

If we have enough time to show multiple rounds or multiple failure modes while preserving old performance, the stricter continual-learning label becomes stronger.

---

# 3. What We Are Building

## We are building

A system that:

1. Generates demonstrations in Antioch
2. Trains a small ACT policy
3. Evaluates it in closed-loop simulation
4. Deploys it to a real SO-101
5. Records the physical rollout
6. Uses an embodied reasoning video critic to analyze real failure
7. Converts that failure into simulation parameters
8. Generates a targeted simulation curriculum
9. Collects corrective demonstrations
10. Retrains ACT
11. Measures before/after performance
12. Redeploys the improved policy

## We are not building

- a giant VLA
- a new foundation model
- a world action model from scratch
- a full online RL system
- an Isaac installation on our laptop
- a robotics stack in TypeScript
- a complicated autonomous coding agent
- a general-purpose simulator-generation system

Keep the learned component small and make the **closed loop** the ambitious part.

---

# 4. Why the Critic Is Only Used in Real Life

In simulation we already have privileged state.

Isaac can directly tell us things such as:

- object pose
- gripper pose
- whether the object was lifted
- whether the object entered the tray
- contacts
- collisions
- timing
- joint state
- success / failure gates

Using a video model to guess these things in simulation would throw away information we already have.

Therefore:

```text
SIMULATION
    ↓
physics / privileged-state evaluator
```

while:

```text
REAL WORLD
    ↓
video
    ↓
embodied reasoning critic
```

The critic's job is specifically:

> **Translate messy real-world behavior into a structured failure signature that the simulator can act on.**

This creates a clean architectural boundary.

---

# 5. Final Architecture

```text
                         INITIAL DATA
                      scripted expert
                       or teleop demos
                              │
                              ▼
                        ┌───────────┐
                        │  ACT v0   │
                        └─────┬─────┘
                              │
                              ▼
                 ┌───────────────────────┐
                 │   ANTIOCH / ISAAC     │
                 │                       │
                 │ randomized rollouts   │
                 │ physics evaluation    │
                 │ held-out evaluation   │
                 └──────────┬────────────┘
                            │
                            ▼
                       SIM GATE
                    pass threshold?
                      │          │
                     no         yes
                      │          │
            more sim data        ▼
                         ┌────────────────┐
                         │ REAL SO-101    │
                         │ ACT rollout    │
                         └───────┬────────┘
                                 │
                                 ▼
                         wrist-camera video
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ EMBODIED REASONING     │
                    │ VIDEO CRITIC           │
                    │                        │
                    │ stage                  │
                    │ failure                │
                    │ likely causes          │
                    │ confidence             │
                    └───────────┬────────────┘
                                │
                                ▼
                       STRUCTURED JSON
                                │
                                ▼
                    ┌────────────────────────┐
                    │ REAL→SIM MAPPER        │
                    │ validated whitelist    │
                    └───────────┬────────────┘
                                │
                                ▼
                       new sim distribution
                                │
        ┌───────────────────────┼──────────────────────┐
        │                       │                      │
     friction               pose noise              mass
     camera pose            actuator lag            object yaw
        │                       │                      │
        └───────────────────────┼──────────────────────┘
                                │
                                ▼
                       ANTIOCH BATCH RUNS
                                │
                                ▼
                    corrective expert demos
                                │
                                ▼
                     aggregate training set
                                │
                                ▼
                           ACT v1
                                │
                                ▼
                        SIM RE-EVALUATION
                                │
                                ▼
                           REDEPLOY
```

---

# 6. Two Feedback Loops

## Inner loop: Simulation improvement

```text
ACT
 ↓
randomized sim
 ↓
physics evaluator
 ↓
weak region
 ↓
targeted expert demonstrations
 ↓
retrain
 ↓
ACT'
```

This loop is cheap and scalable.

## Outer loop: Real-world improvement

```text
ACT'
 ↓
real robot
 ↓
real failure
 ↓
video critic
 ↓
failure diagnosis
 ↓
real→sim mapping
 ↓
new simulation curriculum
 ↓
corrective demonstrations
 ↓
retrain
 ↓
ACT''
```

This is the main project contribution and demo story.

---

# 7. Policy Choice

## Chosen policy

**ACT**

Why:

- supported directly by the hackathon guide
- small enough to train during the event
- works with SO-101
- action chunking is appropriate for manipulation
- easy to compare v0 vs v1
- avoids giant-model setup risk

## Input

Approximately:

```text
wrist RGB
+
joint state
+
task identity
```

For the base hackathon implementation, task identity can remain fixed:

```text
"Put the green cube in the blue tray."
```

## Output

Joint action targets / action chunks.

## Stretch

If time remains, add a tiny amount of task conditioning:

```text
"put cube in left tray"
"put cube in right tray"
```

Do not make this required for the main loop.

---

# 8. Task Design

## Primary task

**Green cube → blue tray**

Why this is the best choice:

- official guides already support it
- simulator setup already exists
- success gates already exist
- real props are expected
- scripted expert path exists
- fastest route to an end-to-end result

## Need a deterministic real-world failure

The demo depends on ACT v0 failing in a repeatable way.

Possible controlled OOD conditions:

- cube shifted farther right
- cube shifted farther left
- different yaw
- slightly larger / smaller object
- more difficult grasp point
- altered surface friction
- changed object mass if practical
- slight camera / pose offset

Choose **one** condition that:

1. v0 fails reliably
2. v1 can be trained to handle
3. is visually obvious to judges
4. is safe for the robot

Do not rely on a random failure.

---

# 9. The Critic

## Preferred interface

The rest of the system should not depend on one vendor-specific API.

Define:

```python
diagnosis = critic.analyze(
    video_path="real_rollout.mp4",
    task="Put the green cube in the blue tray."
)
```

Return a structured schema.

## FailureDiagnosis schema

```json
{
  "success": false,
  "stage": "transport",
  "failure": "object_slip",
  "confidence": 0.91,
  "estimated_causes": [
    {
      "cause": "low_friction",
      "confidence": 0.72
    },
    {
      "cause": "grasp_offset",
      "confidence": 0.21
    }
  ],
  "recommended_sim_changes": [
    {
      "parameter": "object_friction",
      "min": 0.20,
      "max": 0.50
    },
    {
      "parameter": "grasp_pose_noise_mm",
      "min": 0,
      "max": 8
    }
  ]
}
```

## Stage vocabulary

Keep it constrained:

```text
approach
grasp
lift
transport
place
complete
unknown
```

## Failure vocabulary

Examples:

```text
missed_object
bad_alignment
failed_grasp
object_slip
premature_release
collision
unreachable_pose
placement_error
unknown
```

## Why structured output matters

Do **not** do:

```text
model prose
   ↓
arbitrary Python
   ↓
Isaac
```

Do:

```text
model
 ↓
validated JSON
 ↓
whitelist
 ↓
mapper
 ↓
safe simulator parameters
```

---

# 10. Real→Sim Mapper

The mapper is one of the most important pieces.

Its job is not to “understand robotics.”

Its job is to safely convert a diagnosis into a limited set of simulation perturbations.

## Allowed parameters

Start with a tiny whitelist:

```text
object_friction
object_mass
object_x
object_y
object_yaw
grasp_pose_noise
camera_pose_noise
action_delay
joint_target_noise
```

Only add parameters we can reliably change in Antioch.

## Example mapping rules

```text
object_slip
    ↓
decrease friction
increase mass range
add grasp-offset noise
```

```text
failed_grasp
    ↓
increase object pose diversity
increase yaw diversity
add grasp-position noise
```

```text
placement_error
    ↓
randomize tray pose
increase target-position diversity
```

## Important rule

The critic may recommend parameters, but the mapper decides what is legal.

Clamp everything.

---

# 11. Simulation Curriculum

The big idea is:

> One real failure should create many useful simulated experiences.

Example:

```text
ONE REAL FAILURE
        ↓
"object slip"
        ↓
curriculum:
friction 0.20–0.50
yaw ±20°
mass 0.8–1.3x
grasp noise 0–8 mm
        ↓
30 targeted scenarios
        ↓
corrective expert trajectories
```

This is the data flywheel.

## Curriculum modes

### Baseline distribution

Used for initial ACT training.

### Hard held-out distribution

Used only for evaluation.

### Failure-targeted distribution

Generated from real-world diagnosis.

Never report improvement only on the newly generated training set.

Always keep a held-out set.

---

# 12. Metrics

At minimum measure:

## Policy metrics

```text
baseline sim success
held-out sim success
failure-targeted success
real-world success
```

## Example dashboard numbers

```text
                 ACT v0        ACT v1

Baseline sim       90%           93%
Hard set           42%           81%
Real condition     FAIL          PASS
```

## Critic metrics

If practical:

```text
failure stage
critic confidence
diagnosis latency
```

## Data metrics

```text
initial demonstrations:        30
new targeted demonstrations:   20
final dataset:                 50
```

Avoid fake precision.

Only display numbers actually measured.

---

# 13. Technical Stack

## Frontend

**Next.js**

Use it for:

- presentation
- control surface
- live state
- video
- architecture visualization
- metrics
- demo mode
- replay mode

Recommended:

```text
Next.js
React
Tailwind
```

Optional:

```text
Framer Motion
```

if already comfortable with it.

Do not spend time learning a new UI stack at the hackathon.

## Robotics backend

**Python**

Recommended:

```text
FastAPI
Antioch CLI / Python
LeRobot
PyTorch
critic SDK
```

## Architecture

```text
localhost:3000
NEXT.JS
    │
    │ HTTP + SSE/WebSocket
    ▼
localhost:8000
FASTAPI
    │
    ├── Antioch
    ├── ACT
    ├── LeRobot
    ├── SO-101
    ├── Critic
    └── Real→Sim mapper
```

The Next.js app should never directly control robot motors.

---

# 14. Suggested Repository Structure

```text
physical-ai-sprint/
│
├── frontend/
│   ├── app/
│   ├── components/
│   │   ├── RobotVideo.tsx
│   │   ├── LearningLoop.tsx
│   │   ├── PolicyMetrics.tsx
│   │   ├── FailureDiagnosis.tsx
│   │   ├── CurriculumPanel.tsx
│   │   └── DemoControls.tsx
│   └── lib/
│
├── backend/
│   ├── main.py
│   ├── critic.py
│   ├── schemas.py
│   ├── mapper.py
│   ├── orchestrator.py
│   ├── antioch_client.py
│   ├── policy.py
│   ├── robot.py
│   └── state.py
│
├── sim/
│   ├── scenarios/
│   ├── curriculum.py
│   └── eval.py
│
├── training/
│   ├── build_dataset.py
│   ├── train_act.sh
│   └── checkpoints/
│
├── artifacts/
│   ├── videos/
│   ├── diagnoses/
│   ├── evals/
│   └── demo/
│
└── README.md
```

Do not reorganize the official Antioch starter repo so aggressively that its tooling breaks.

If Antioch gives us an existing repository, adapt this structure around it.

---

# 15. Backend API

Keep the UI/backend contract tiny.

## Status

```http
GET /api/status
```

Example:

```json
{
  "stage": "diagnosing",
  "policy": "v0",
  "environment": "real",
  "run_id": "abc123"
}
```

## Start sim

```http
POST /api/sim/run
```

## Deploy

```http
POST /api/deploy
```

## Analyze failure

```http
POST /api/critic/analyze
```

## Generate curriculum

```http
POST /api/curriculum/generate
```

## Run targeted batch

```http
POST /api/sim/batch
```

## Load policy

```http
POST /api/policy/load
```

## Replay

```http
POST /api/demo/replay
```

## Event stream

Use SSE or WebSocket:

```text
sim_started
sim_passed
deploy_started
real_failed
video_ready
critic_started
diagnosis_ready
curriculum_ready
batch_started
batch_progress
policy_loaded
redeploy_started
real_success
```

This makes the Next.js dashboard mostly an event renderer.

---

# 16. Dashboard Design

## Design goal

The UI should make the system understandable in **10 seconds**.

It should not look like:

```text
terminal
terminal
terminal
terminal
```

It should look like a robotics product.

## Permanent header

```text
CONTINUAL EMBODIED LEARNING

SO-101 · ACT · Antioch · Embodied Critic

Policy v1                              SYSTEM LIVE
```

## Main learning loop

Always visible:

```text
SIMULATE
   ↓
VERIFY
   ↓
DEPLOY
   ↓
EXPERIENCE
   ↓
DIAGNOSE
   ↓
ADAPT
   ↓
LEARN
   ↺
```

Highlight the active stage.

## Main video area

Large.

Show either:

- Antioch live stream
- real robot live / latest recorded video
- replay

Overlay:

```text
ENVIRONMENT: REAL
POLICY: ACT v0
TASK: cube → tray
STATUS: FAILURE
```

## Failure panel

```text
REAL-WORLD FAILURE

APPROACH       ✓
GRASP          ✓
LIFT           ✓
TRANSPORT      ✗
PLACE          -

Object slipped during transport

Critic confidence: 91%
```

## Reality → sim panel

```text
REALITY → SIM

Detected weakness

friction mismatch       72%
grasp offset            21%
other                    7%

New curriculum

friction       0.20 ───── 0.50
pose noise     ±8 mm
mass           0.8x ───── 1.3x
```

## Batch simulation panel

```text
TARGETED ANTIOCH RUNS

01 ✓
02 ✓
03 ✗
04 ✓
05 ✓
06 ✗
...
```

If we can show multiple little thumbnails, great.

If not, show a progress grid.

## Improvement panel

```text
POLICY IMPROVEMENT

              v0          v1

Baseline      90%    →    93%
Hard set      42%    →    81%
Real          FAIL   →    PASS
```

## Final state

Make it visually simple:

```text
THE ROBOT LEARNED FROM DEPLOYMENT

ACT v0  →  ACT v1

REAL FAIL  →  REAL PASS
```

---

# 17. Science-Fair Demo Requirement

Round 1 is a science-fair-style format.

Judges circulate between stations.

Therefore the primary demo cannot require five uninterrupted minutes.

We need:

## Mode A: 60–90 second booth walkthrough

Repeatable many times.

## Mode B: instant replay

A judge can walk up at any point and understand the system.

## Mode C: live robot

Available when hardware is free and conditions are right.

## Mode D: 4–5 minute finalist presentation

Only needed if selected for the final six.

---

# 18. Science-Fair Booth Demo — 60 to 90 Seconds

## 0–10 sec: thesis

Dashboard already shows:

```text
SIM ✓ → REAL ✗ → DIAGNOSE → ADAPT → LEARN → REAL ✓
```

Say:

> “Most robot policies stop learning once they're deployed. Ours uses deployment failures as the next training signal.”

## 10–25 sec: show failure

Play a 5–10 second real SO-101 failure.

```text
ACT v0
REAL
FAIL
```

Say:

> “The policy passed simulation, but reality exposed a weakness.”

## 25–45 sec: critic

Show:

```text
approach       PASS
grasp          PASS
lift           PASS
transport      FAIL

object slip
```

Say:

> “Our embodied reasoning critic watches the real rollout and turns the failure into a structured diagnosis.”

## 45–65 sec: reality → simulation

Show the diagnosis changing the simulator distribution.

```text
friction
0.60 → 0.20–0.50

grasp noise
0 → ±8 mm
```

Show targeted Antioch runs.

Say:

> “That one physical failure generates an entire corrective curriculum in simulation.”

## 65–90 sec: payoff

Show:

```text
ACT v0 hard set: 42%
ACT v1 hard set: 81%

REAL:
v0 FAIL → v1 PASS
```

Say:

> “The deployment failure became the robot's next training data.”

Done.

---

# 19. Finalist Demo — Approximately 5 Minutes

## 0:00–0:35 — thesis

> “Robots today are usually trained, evaluated, and then frozen. We built a loop where deployment itself becomes the next training phase.”

Show architecture.

## 0:35–1:15 — simulation

Show ACT v0 passing in Antioch.

```text
SIM VERIFIED
27 / 30
```

> “Once the policy passes our physics-based simulation gate, we let it touch the real robot.”

## 1:15–2:00 — live real deployment

Run ACT v0 on the controlled hard condition.

Ideally it fails.

Dashboard:

```text
REAL DEPLOYMENT
FAIL
```

## 2:00–2:45 — live critic

Send the just-recorded rollout.

Show:

```text
approach      ✓
grasp         ✓
lift          ✓
transport     ✗
```

Then diagnosis.

## 2:45–3:30 — adapt simulation

Show:

```text
REALITY → SIM
```

Generate targeted scenario parameters.

Launch a small Antioch batch live if latency permits.

## 3:30–4:05 — learning result

Do **not** train ACT for an hour in front of judges.

Use a checkpoint already trained using the exact loop.

Be transparent:

> “The retraining step takes longer than the live presentation, so this v1 checkpoint was trained beforehand from the curriculum you just saw generated.”

Show actual measured v0 vs v1 metrics.

## 4:05–4:50 — redeploy

Load v1.

Run the same physical condition.

Robot succeeds.

## 4:50–5:00 — close

> **“The robot didn't just fail. It turned that deployment experience into its next training curriculum.”**

Final screen:

```text
ACT v0       ACT v1
FAIL    →    SUCCESS

THE ROBOT LEARNED FROM DEPLOYMENT
```

---

# 20. What Must Be Live vs What Can Be Precomputed

## Ideally live

- Next.js dashboard
- Antioch sim stream
- physical robot attempt
- recording of rollout
- critic request
- structured diagnosis
- reality→sim mapper
- launching at least a tiny simulation batch
- loading v1
- final physical attempt

## Precompute

- ACT v0 training
- ACT v1 training
- large evaluation batches
- polished before/after metrics
- backup real-world failure video
- backup real-world success video

## Never pretend precomputed training happened live

Just say:

> “We precomputed the training step because it takes longer than the demo; the diagnosis and curriculum generation you're seeing are live.”

That is completely reasonable.

---

# 21. Demo Reliability

A hackathon demo should have multiple layers of fallback.

## Level 1

Full live loop.

## Level 2

Live robot failure + live critic + precomputed v1.

## Level 3

Recorded robot failure + live critic + live curriculum generation.

## Level 4

Entire real-world sequence replayed, with live dashboard narration.

The packet explicitly allows live **or recorded** demos during the science-fair round.

## Dashboard controls

Include hidden or unobtrusive controls:

```text
RUN LIVE
LOAD RECORDED FAILURE
LOAD RECORDED SUCCESS
RESET DEMO
LOAD V0
LOAD V1
```

Do not let a dead USB cable destroy the story.

---

# 22. What to Set Up Right Now

Do these before implementing novel logic.

## 1. Antioch access

Need:

- team invite
- starter project
- authentication working

From the starter project:

```bash
cd ~/my-sim
uv sync
.venv/bin/antioch auth login
.venv/bin/antioch --help
```

Then run **one supplied scenario unchanged**.

Success means:

```text
laptop
  ↓
Antioch
  ↓
cloud Isaac
  ↓
browser simulation
```

Do not modify the sim until this works.

## 2. Separate LeRobot environment

```bash
conda create -n lerobot python=3.10 ffmpeg -c conda-forge -y
conda activate lerobot
pip install lerobot opencv-python
```

Verify:

```bash
python -c "import lerobot, torch; print(lerobot.__version__); print(torch.backends.mps.is_available())"
```

Keep it separate from Antioch's project environment.

```text
my-sim/.venv
    ↓
simulation

conda env: lerobot
    ↓
training + hardware
```

## 3. SO-101 calibration

As soon as hardware is available:

- identify leader/follower ports
- calibrate both
- back up calibration data
- test teleoperation
- record one short episode
- test dry-run path

Do this early because hardware queue is the biggest external dependency.

## 4. Critic credentials

Get the API key ready.

Verify:

```text
text request works
↓
short video request works
↓
structured JSON works
```

Do this before relying on the model during the demo.

## 5. Next.js skeleton

Build the UI before the robotics loop is finished.

Need only:

- page shell
- learning-loop component
- big video component
- failure panel
- curriculum panel
- metrics panel
- demo controls

Use dummy JSON first.

## 6. FastAPI skeleton

Create endpoints with mock responses.

Make this work:

```text
Next.js
   ↓
FastAPI
   ↓
dummy diagnosis
   ↓
dashboard updates
```

before connecting the real robot.

## 7. Critic schema

Freeze the `FailureDiagnosis` JSON contract early.

## 8. Mapper whitelist

Freeze the handful of sim parameters we actually support.

---

# 23. Phase Plan for Hackathon Day

Official hacking window:

```text
10:30 → 15:30
```

We have only about five hours.

## Before 10:30, if allowed

Do as much as possible:

- accounts
- API credentials
- Next.js skeleton
- FastAPI skeleton
- dependency installation
- repo setup
- dashboard styling

## 10:30–11:00 — prove plumbing

Parallelize.

### Sim person

- Antioch auth
- run starter scenario
- find SO-101 scenario
- verify browser stream

### Hardware person

- SO-101 calibration
- teleop test
- camera test

### ML person

- confirm dataset format
- run scripted expert
- start initial data generation

### Product/demo person

- Next.js dashboard
- backend event stream
- dummy demo state machine

### Critic person

- video API test
- structured output
- failure schema

## 11:00–11:30 — initial data

Use scripted expert unless teleop is already perfect.

Goal:

```text
~30 good sim episodes
```

Start ACT training immediately.

## 11:30–12:30 — training + integration

While ACT trains:

- integrate critic
- build mapper
- parameterize failure scenarios
- connect dashboard to backend
- create replay mode
- identify deterministic real-world failure condition

## 12:30–13:15 — v0 evaluation

Run:

- baseline eval
- hard held-out eval

Pick best checkpoint.

Do not choose based only on training loss.

## 13:15–13:45 — first real deployment

- dry-run
- real SO-101
- capture several episodes
- deliberately test hard condition
- save at least one good failure video

## 13:45–14:15 — critic + mapper

- analyze real failure
- produce diagnosis
- map to sim perturbations
- generate targeted scenarios

## 14:15–14:45 — targeted data

Generate corrective trajectories.

If training v1 will take too long:

- start immediately
- reduce steps if justified
- keep v1 training while demo UI is polished

## 14:45–15:10 — evaluation

Compare:

```text
v0
vs
v1
```

on the same held-out hard set.

Try real v1 deployment if checkpoint is ready.

## 15:10–15:30 — stop building

Only:

- lock demo state
- collect backup videos
- verify replay mode
- verify metrics
- clean dashboard
- rehearse 60-second pitch
- rehearse 5-minute pitch

No new features after 15:10 unless the demo is broken.

---

# 24. Team Split

For 4 people:

## Person 1 — Simulation / Antioch

Own:

- scenarios
- parameterization
- batch runs
- physics eval
- sim artifacts

## Person 2 — Policy / LeRobot

Own:

- dataset
- ACT training
- checkpoints
- real robot policy execution

## Person 3 — Critic / Mapper

Own:

- real video ingestion
- critic
- schema
- mapper
- targeted curriculum

## Person 4 — Product / Integration

Own:

- Next.js
- FastAPI
- state machine
- demo orchestration
- metrics
- replay mode

For 5 people, split hardware from policy:

## Person 5 — Hardware

Own:

- SO-101 calibration
- cameras
- dry-runs
- safety
- repeatable real conditions
- recording

---

# 25. Safety

Non-negotiable:

- dry-run before live robot execution
- hand near power switch on first runs
- do not cut torque while holding a commanded pose
- return safely to rest before disconnect
- pin driver / LeRobot / calibration versions
- assert units
- do not bypass safety wrappers
- use controlled failure conditions, not dangerous ones

Our goal is to make the **task fail**, not the hardware.

---

# 26. Risk Matrix

| Risk | Mitigation |
|---|---|
| Hardware unavailable | Build almost everything in sim; use recorded real rollout |
| Hardware queue | Get calibration and failure videos as early as possible |
| ACT does not transfer | Demo critic + real→sim adaptation; emphasize sim-gap diagnosis |
| ACT v1 training too slow | Precompute smaller v1 / show targeted sim improvement |
| Critic API fails | Fallback video-capable model behind same interface |
| Critic output is noisy | Constrained schema + whitelist mapper |
| Critic diagnosis is wrong | Treat as hypothesis; simulator validates it |
| Antioch cold-start issues | Warm machine early and preserve working setup |
| Real v0 unexpectedly succeeds | Use a stronger but safe deterministic OOD condition |
| Real v1 unexpectedly fails | Show repeated quantitative sim improvement + recorded successful real v1 |
| Dashboard breaks | Static replay JSON + prerecorded videos |
| Internet instability | Cache all possible assets and failure/success videos locally |

---

# 27. MVP / Strong / Killer Demo

## MVP

- ACT v0 runs in Antioch
- closed-loop sim eval
- ACT v0 runs on SO-101
- real rollout video captured
- critic produces structured diagnosis
- mapper produces new sim parameters
- targeted Antioch scenario launches
- Next.js dashboard visualizes the entire loop

## Strong

Everything above plus:

- corrective expert data generated
- ACT v1 trained
- v0 vs v1 measured on same held-out set
- meaningful quantitative improvement

## Killer

Everything above plus:

```text
ACT v0
REAL FAIL
   ↓
critic
   ↓
new simulation curriculum
   ↓
ACT v1
   ↓
same real condition
   ↓
REAL SUCCESS
```

That is the outcome to optimize for.

---

# 28. Judge Story

## Perception

The embodied critic interprets the real-world rollout.

## Reasoning

The system decides why the physical deployment failed and converts that into a new simulation curriculum.

## Action

ACT controls the SO-101.

## Ambition

The project spans:

```text
simulation
training
evaluation
real hardware
video reasoning
real-to-sim adaptation
retraining
redeployment
```

## Functionality

Use the official SO-101 / Antioch / ACT infrastructure wherever possible.

## Creativity

The critic is not just judging success.

It is **closing the real-to-sim learning loop**.

## Architectural quality

Each component has one clear job:

```text
ACT
actions

Isaac
physics + ground truth

Real robot
exposes sim gap

Critic
interprets real failure

Mapper
turns diagnosis into legal parameters

Antioch
scales corrective experience

Next.js
control + visualization
```

---

# 29. Pitches

## 5-second

> **A robot that learns from what happens after deployment.**

## 15-second

> “We train a manipulation policy in simulation, deploy it to the real SO-101, and when reality exposes a failure, an embodied reasoning critic turns that failure into a new simulation curriculum. The policy retrains and gets redeployed.”

## 30-second

> “Most robot policies are trained and then frozen. We built a real-to-sim-to-real learning loop. ACT learns the manipulation task in Antioch, passes a physics-based simulation gate, and gets deployed to the SO-101. When the real robot fails, a video reasoning critic diagnoses the physical failure and converts it into targeted domain randomization. Antioch generates corrective experience, we retrain the policy, and redeploy it.”

## Closing line

> **The robot didn't just fail. It turned that deployment experience into its next training curriculum.**

Alternative:

> **Deployment isn't the end of training. It's the next data collection phase.**

Alternative:

> **Reality tells us what simulation forgot.**

---

# 30. Questions Judges May Ask

## “Why use a critic only in the real world?”

Because in simulation we have exact privileged state and physics-based success gates. The critic is valuable where that privileged information disappears: the real robot.

## “Why not just domain-randomize everything from the beginning?”

Because uniform randomization wastes compute and data on irrelevant variations.

The real failure tells us **where the current policy actually needs more coverage**.

## “How do you know the critic's explanation is correct?”

We treat the diagnosis as a hypothesis, not ground truth.

The simulator tests it.

If targeted randomization improves the policy on the corresponding failure set, the hypothesis produced useful training data.

## “Is the model directly controlling the robot?”

No.

The learned manipulation policy controls the robot.

The reasoning critic only analyzes completed physical rollouts.

## “Why ACT instead of a VLA?”

A small policy lets us demonstrate the complete learning lifecycle in a single hackathon rather than spending the event fine-tuning a huge model.

## “Is retraining live?”

The curriculum generation can be live.

For a short presentation, the expensive retraining checkpoint should be prepared ahead of time using the same generated curriculum.

## “What's novel?”

Do not claim we invented continual robot learning.

The project contribution is the **integrated real-to-sim-to-real learning loop**:

```text
physical failure
→ embodied diagnosis
→ targeted sim curriculum
→ policy improvement
→ redeployment
```

## “What scales?”

A single physical failure can generate many targeted simulation scenarios and corrective trajectories.

That is the data flywheel.

---

# 31. Final Checklist

## Before hacking

- [ ] Antioch invite received
- [ ] starter project cloned / available
- [ ] `uv sync` works
- [ ] Antioch authentication works
- [ ] one starter scenario runs
- [ ] browser stream works
- [ ] LeRobot environment created
- [ ] critic API key works
- [ ] short video analysis works
- [ ] Next.js app runs
- [ ] FastAPI app runs
- [ ] Next.js receives dummy backend events

## Hardware

- [ ] leader port known
- [ ] follower port known
- [ ] calibration complete
- [ ] calibration backed up
- [ ] teleop works
- [ ] cameras work
- [ ] dry-run works
- [ ] safe rest behavior confirmed

## Sim

- [ ] scripted expert runs
- [ ] episodes save
- [ ] physics success gates work
- [ ] scenario parameters are configurable
- [ ] batch execution works
- [ ] held-out set exists

## Policy

- [ ] ACT training starts
- [ ] checkpoints save
- [ ] sim eval works
- [ ] v0 selected
- [ ] real execution works
- [ ] v1 training path works

## Critic

- [ ] real video uploads
- [ ] structured diagnosis validates
- [ ] stage vocabulary constrained
- [ ] failure vocabulary constrained
- [ ] mapper only accepts whitelist

## Dashboard

- [ ] learning-loop visualization
- [ ] live/replay video
- [ ] current policy version
- [ ] environment indicator
- [ ] failure panel
- [ ] curriculum panel
- [ ] batch progress
- [ ] before/after metrics
- [ ] replay mode
- [ ] reset demo button

## Demo assets

- [ ] best v0 sim video
- [ ] best v0 real failure
- [ ] diagnosis JSON
- [ ] curriculum JSON
- [ ] targeted sim batch results
- [ ] v0 metrics
- [ ] v1 metrics
- [ ] best v1 real success
- [ ] backup screen recording

## Science fair

- [ ] 60-second version rehearsed
- [ ] 90-second version rehearsed
- [ ] dashboard understandable without narration
- [ ] live robot optional, not required to explain project
- [ ] result visible permanently on screen

## Finalist stage

- [ ] 5-minute script rehearsed
- [ ] exact transitions known
- [ ] no live hour-long training
- [ ] one person talks
- [ ] one person operates
- [ ] backup replay ready

---

# 32. North Star

Every engineering decision should optimize for this sequence:

```text
SIM PASS
   ↓
REAL FAIL
   ↓
UNDERSTAND WHY
   ↓
CHANGE SIM
   ↓
LEARN
   ↓
REAL PASS
```

If a feature does not make that loop more reliable, more measurable, or easier for judges to understand, it is probably not worth building during the sprint.

**The project is not the ACT model.  
The project is the learning loop.**
