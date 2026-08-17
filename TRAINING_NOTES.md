# ACT training playbook (researched 2026-08-17)

Source-verified against LeRobot v0.4.4 code and SO-101 case studies. The five
decisions below make or break the day — details and citations follow.

## The five that decide the day

1. **Do NOT use `chunk_size=100, n_action_steps=100` at 10 Hz.** That is a
   10-second open-loop policy (~1.4 decisions per episode) — it physically
   cannot react to a moved block, which kills our whole OOD story. Use
   `--policy.chunk_size=30 --policy.n_action_steps=15`. But do not overshoot:
   `n_action_steps=1` scores 2% in LeRobot's own measurements. chunk_size is
   baked into the checkpoint — decide before v0.
2. **Units at real-robot deploy: our data is radians; LeRobot's SO-101 driver
   defaults to DEGREES for joints and 0–100 for the gripper.** Verified in
   v0.4.4 source. Build a conversion adapter and validate it by REPLAYING one
   dataset episode on the real arm before ever running the policy. Set
   `--robot.max_relative_target` small on first runs.
3. **Two cameras.** Controlled SO-101 ablation: ACT wrist-only 20% vs
   wrist+side 50%. Our sim already has a side camera — record it into the
   dataset as `observation.images.front`.
4. **v1 = retrain from scratch on base + corrective MERGED** (corrective
   ≈20–35% of the mix; duplicate corrective 2× if needed). The documented
   SO-ARM101 failure: fine-tuning on corrections alone made the policy "go to
   the edge" for everything; the merged retrain hit 90%. Merge with
   `lerobot-edit-dataset --operation.type merge` (task strings must match —
   ours currently differ between build_dataset.py and the plan; unify first).
5. **Evaluate by closed-loop rollouts, never by loss** (loss-vs-performance
   correlation is r≈0.3). Sim rollouts are cheap: 50 per condition, identical
   seeds for v0 and v1 (LeRobot seeds deterministically), report Wilson CIs.
   `--eval.batch_size=1` is mandatory with Isaac (default 50 spawns 50 sims
   and crashes).

## Also do (cheap, high value)

- **DART noise in the scripted expert**: add small Gaussian noise to commanded
  targets during demo collection (never on the gripper channel, faded near
  waypoints) and randomize the start pose — the policy learns recovery, which
  is literally our pitch. Record the NOISY COMMANDED action as `action`.
- **60–100 episodes, stratified into spawn bins** (10+ per bin), not 30
  uniform. The best SO-101 ACT result (90% ID / 75% OOD) came from binned
  coverage; 50 unbinned gave 60%/10%.
- `--dataset.image_transforms.enable=true` (OFF by default; NOT inherited by
  fine-tunes). `--policy.use_amp=false` on MPS.
- Check `meta/stats.json` after building: any action/state std < 1e-3 (e.g. a
  joint that never moves) explodes normalization — switch that path to MIN_MAX.
- Time 200 training steps first; if < 4 steps/s on the Mac, rent a GPU
  (A100 does this run in well under an hour). 15k steps is the floor, 40–60k
  is where ACT gets good; save every 5k and closed-loop-eval each checkpoint.
- Don't use `--dataset.eval_split` (open LeRobot bug: policies freeze).
- Expected honest numbers: ID→OOD drops of 40–90 points are NORMAL published
  behavior for ACT under spatial shift — that's our engineered failure working,
  not a bug. Scripted demos beat human demos (+36 pts in the ACT paper).

## The slide sentence to aim for

> "v1: 34/50 on the held-out OOD set vs v0: 12/50, paired on identical seeds;
> both ~48/50 in-distribution — the corrective data fixed the hard region
> without regressing the easy one."

That last clause is what a knowledgeable judge looks for.

## Sim curriculum for v1 (strongest version of our story)

Log the states where v0 fails, reset the sim to those states, let the scripted
expert solve from there, record those as the corrective episodes. That is
textbook DAgger data aggregation, gated by the critic's diagnosis.
