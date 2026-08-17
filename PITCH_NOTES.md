# Pitch notes: related work & judge Q&A ammo

Researched 2026-08-17. Use these when judges ask "is this novel?" or "what's this
related to?" — naming the prior art ourselves, before they do, is the power move.

## The honest novelty claim (rehearse this verbatim)

> "First system we're aware of that closes the loop from a structured VLM failure
> diagnosis to a targeted domain-randomization curriculum and back onto the real
> arm. Each half is well-established — happy to name the papers — but we found no
> prior work joining them."

Do NOT claim "first VLM robot critic" or "novel adaptive domain randomization" —
both would lose in one follow-up question.

## The three talking points

1. **Two mature literatures, never joined.** The failure-diagnosis lineage
   (REFLECT, AHA, RoboFAC, Gemini ER 2 itself) always ends in a runtime fix:
   re-plan, retry, nudge. The sim-adaptation lineage (SimOpt, OpenAI's ADR,
   AdaptSim) always consumes a scalar "sim ≠ real" signal — it never knows *why*.
   We are the connective tissue: a semantic diagnosis **compiled** into a
   parametric curriculum. Call the mapper a *compiler*: "we compile a failure
   into a curriculum."

2. **LLM-guided domain randomization exists (DrEureka, RSS 2024) — but it's
   open-loop.** DrEureka's LLM picks randomization ranges entirely in sim,
   before deployment, for RL. Ours is conditioned on an observed real failure,
   on real hardware. One line: *"DrEureka with a real-world error signal,
   feeding imitation learning instead of RL."*

3. **The deterministic mapper is the architecture, not a shortcut.** The VLM is
   untrusted: closed-enum schema in, clamped ranges out, no model-generated code
   near the physics engine. Answer to "what if the VLM hallucinates?": a bad
   diagnosis costs one wasted sim batch, never an unsafe real action — and the
   dashboard shows the clamp firing.

## Names judges may know (one-line contrasts)

- **REFLECT / RoboFail** (CoRL 2023) — VLM explains failures, then re-plans the
  same episode. Ours fixes the *weights*, not the plan.
- **AHA** (NVIDIA, ICLR 2025) — perturbs sim to train a failure critic. We
  invert it: the critic's verdict decides how to perturb sim.
- **Gemini Robotics ER 2** (July 2026) — Google ships it as a runtime
  orchestrator; we use it as a *training-signal generator*. (Reframes their
  product as our component.)
- **SimOpt** (NVIDIA, ICRA 2019) — canonical "real rollouts tune sim params,"
  but its signal is a trajectory error metric. *"SimOpt where the residual is
  language."*
- **DrEureka / Eurekaverse** (2024) — see talking point 2.
- **RialTo** (RSS 2024) — real-to-sim-to-real, but rebuilds *geometry*; we
  retune *distributions*, triggered by one specific failure.
- **DAgger** (2011) / **DART** (2017) — our corrective-demo half. Every DAgger
  descendant gates on policy uncertainty; we gate on the critic's causal
  diagnosis — "semantic DAgger."
- **PatchWork** (StarkHacks 2026, closest hackathon analog — SO-101 + ACT +
  VLM + rule-based failure patcher) — patches runtime motion offsets; the
  policy never improves. Ours changes the policy. Sharpest demo-vs-demo
  contrast; name it first.

## Numbers worth quoting

- A published SO-101 VLA benchmark (arXiv 2606.08881) measured ACT's failure
  **recovery rate at 6.5%** on this exact hardware — the baseline our loop
  attacks.
- Practitioner reports of RL fine-tuning on SO-101 (HIL-SERL writeups): 3+ hours
  of human babysitting for ~70% grasp. Our loop's cost: **one failure video**.
- SimOpt needed 3–5 real iterations; RialTo ~15 demos + RL. Ours: one video.

## If time permits (highest-value additions, in order)

1. **The decisive ablation:** targeted curriculum vs uniform randomization at
   the same sim budget. If targeted wins, the critic is proven load-bearing.
   This is the strongest possible slide.
2. **Held-out OOD bin protocol:** hold out one spatial bin of block positions
   as the eval set (standard in ACT-on-SO-101 writeups); report in-distribution
   vs held-out success for v0 and v1.
3. **Graded outcome score** (1–5: no success / minimal / partial / near /
   perfect, from RoboReward) — shows v0→v1 movement even before binary success
   flips.
4. **"Task-driven adaptation, not system identification"** (AdaptSim's framing):
   we don't claim to match reality's friction number, we maximize real task
   success. Preempts "but is your friction coefficient right?"
