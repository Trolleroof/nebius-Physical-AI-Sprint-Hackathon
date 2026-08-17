/**
 * Events in, dashboard state out. A pure function with no React in it, so
 * the whole demo can be stepped through and tested without a browser.
 *
 * The same reducer serves a replayed fixture and a live SSE stream — that is
 * what makes section 21's fallback levels free rather than extra work.
 */

import {
  FailureDiagnosis,
  LOOP_STAGES,
  LOOP_STAGE_FOR_EVENT,
  LoopStage,
  PolicyMetrics,
  RobotEvent,
  SimChange,
} from "./contract";

export type RailStatus = "pending" | "active" | "done" | "failed";

export interface DashboardState {
  runId: string | null;
  /** Which rail stage is lit right now. */
  activeStage: LoopStage | null;
  rail: Record<LoopStage, RailStatus>;
  /** Increments on redeploy, so the rail can replay its second lap. */
  cycle: number;

  policy: string | null;
  checkpoint: string | null;
  environment: "sim" | "real";
  condition: string | null;
  outcome: "running" | "success" | "failure" | null;

  sim: { completed: number; total: number; successes: number; passed: boolean | null };
  video: { url: string; poster: string | null; duration: number | null } | null;

  criticRunning: boolean;
  diagnosis: FailureDiagnosis | null;
  diagnosisLatencyMs: number | null;

  curriculum: {
    changes: SimChange[];
    baseline: Record<string, number>;
    nScenarios: number;
    unmappableReason: string | null;
  } | null;

  /** One slot per targeted scenario; null until that scenario reports. */
  batch: { results: Array<boolean | null>; total: number; complete: boolean } | null;

  metrics: PolicyMetrics | null;
  error: { message: string; where: string } | null;

  lastSeq: number;
  lastTs: number | null;
}

export function initialState(): DashboardState {
  return {
    runId: null,
    activeStage: null,
    rail: Object.fromEntries(LOOP_STAGES.map((s) => [s, "pending"])) as Record<LoopStage, RailStatus>,
    cycle: 0,
    policy: null,
    checkpoint: null,
    environment: "sim",
    condition: null,
    outcome: null,
    sim: { completed: 0, total: 0, successes: 0, passed: null },
    video: null,
    criticRunning: false,
    diagnosis: null,
    diagnosisLatencyMs: null,
    curriculum: null,
    batch: null,
    metrics: null,
    error: null,
    lastSeq: -1,
    lastTs: null,
  };
}

/**
 * Advance the rail. Everything before the new stage is marked done, unless it
 * already failed — a failed EXPERIENCE must keep reading as a failure once the
 * highlight has moved on to DIAGNOSE, since that failure is the whole story.
 */
function advanceRail(state: DashboardState, next: LoopStage): DashboardState["rail"] {
  const rail = { ...state.rail };
  const target = LOOP_STAGES.indexOf(next);

  LOOP_STAGES.forEach((stage, i) => {
    if (rail[stage] === "failed") return;
    // Everything ahead of the highlight is complete...
    if (i < target) rail[stage] = "done";
    // ...and any stage still flagged active from an earlier lap must stand
    // down, or two stages glow at once once the loop wraps past LEARN.
    else if (stage !== next && rail[stage] === "active") rail[stage] = "done";
  });
  if (rail[next] !== "failed") rail[next] = "active";
  return rail;
}

export function reduce(state: DashboardState, event: RobotEvent): DashboardState {
  // A new run wipes the board. Otherwise the previous run's diagnosis, batch
  // grid and metrics stay on screen underneath the new one — the panels only
  // ever populate, so nothing else would clear them.
  if (state.runId !== null && event.run_id !== state.runId) {
    state = initialState();
  }

  // Out-of-order or duplicate delivery: keep the newer view of the world.
  if (event.seq <= state.lastSeq && state.lastSeq !== -1) return state;

  let next: DashboardState = {
    ...state,
    runId: event.run_id,
    lastSeq: event.seq,
    lastTs: event.ts,
  };

  const railStage = LOOP_STAGE_FOR_EVENT[event.type];
  if (railStage) {
    next.rail = advanceRail(next, railStage);
    next.activeStage = railStage;
  }

  switch (event.type) {
    case "sim_started":
      next.environment = "sim";
      next.policy = event.policy;
      next.outcome = "running";
      next.sim = { completed: 0, total: event.n_episodes, successes: 0, passed: null };
      break;

    case "sim_progress":
      next.sim = { ...next.sim, completed: event.completed, total: event.total, successes: event.successes };
      break;

    case "sim_passed":
      next.sim = { ...next.sim, completed: event.total, total: event.total, successes: event.successes, passed: true };
      break;

    case "sim_failed":
      next.sim = { ...next.sim, successes: event.successes, total: event.total, passed: false };
      next.rail = { ...next.rail, verify: "failed" };
      break;

    case "deploy_started":
    case "redeploy_started": {
      next.environment = "real";
      next.policy = event.policy;
      next.condition = event.condition;
      next.outcome = "running";
      next.video = null;
      if (event.type === "redeploy_started") {
        // Second lap: clear the previous real-world verdict so the rail does
        // not still read FAILED while the improved policy is running.
        next.cycle = state.cycle + 1;
        next.rail = { ...next.rail, experience: "pending", deploy: "active" };
      }
      break;
    }

    case "video_ready":
      next.video = { url: event.url, poster: event.poster_url, duration: event.duration_s };
      break;

    case "real_failed":
      next.outcome = "failure";
      next.policy = event.policy;
      next.rail = { ...next.rail, experience: "failed" };
      break;

    case "real_success":
      next.outcome = "success";
      next.policy = event.policy;
      next.rail = { ...next.rail, experience: "done" };
      break;

    case "critic_started":
      next.criticRunning = true;
      next.diagnosis = null;
      break;

    case "diagnosis_ready":
      next.criticRunning = false;
      next.diagnosis = event.diagnosis;
      next.diagnosisLatencyMs = event.latency_ms;
      break;

    case "curriculum_ready":
      next.curriculum = {
        changes: event.changes,
        baseline: event.baseline,
        nScenarios: event.n_scenarios,
        unmappableReason: event.unmappable_reason ?? null,
      };
      break;

    case "batch_started":
      next.batch = {
        results: Array(event.n_scenarios).fill(null),
        total: event.n_scenarios,
        complete: false,
      };
      break;

    case "batch_progress": {
      const results = next.batch ? [...next.batch.results] : Array(event.total).fill(null);
      results[event.index] = event.success;
      next.batch = { results, total: event.total, complete: false };
      break;
    }

    case "batch_complete":
      if (next.batch) next.batch = { ...next.batch, complete: true };
      break;

    case "policy_loaded":
      next.policy = event.policy;
      next.checkpoint = event.checkpoint;
      break;

    case "metrics_ready":
      next.metrics = event.metrics;
      break;

    case "error":
      next.error = { message: event.message, where: event.where };
      break;
  }

  return next;
}

/** Fold a whole event list — used to seek to a point in a replay. */
export function reduceAll(events: RobotEvent[]): DashboardState {
  return events.reduce(reduce, initialState());
}
