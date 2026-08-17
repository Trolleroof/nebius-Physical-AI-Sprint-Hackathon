/**
 * TypeScript mirror of backend/schemas.py and backend/events.py.
 *
 * These two files are one contract in two languages. If you change a field
 * here, change it there in the same commit — nothing enforces it at build
 * time, and a silent drift shows up as a blank panel on demo day.
 */

// --- critic vocabulary (schemas.py) -----------------------------------------

export const ORDERED_STAGES = ["approach", "grasp", "lift", "transport", "place"] as const;
export type OrderedStage = (typeof ORDERED_STAGES)[number];
export type Stage = OrderedStage | "complete" | "unknown";

export type FailureMode =
  | "missed_object"
  | "bad_alignment"
  | "failed_grasp"
  | "object_slip"
  | "premature_release"
  | "collision"
  | "unreachable_pose"
  | "placement_error"
  | "none"
  | "unknown";

/** The keyword arguments of so101_pick_place — nothing else can be varied. */
export type SimParameter = "block_x" | "block_y" | "tray_x" | "tray_y";

/** Below this the UI says "critic unsure" instead of printing a percentage. */
export const CONFIDENCE_FLOOR = 0.35;

/** Display labels and units. Keep in sync with PARAMETER_BOUNDS in schemas.py. */
export const PARAMETER_META: Record<SimParameter, { label: string; unit: string }> = {
  block_x: { label: "Block X", unit: "m" },
  block_y: { label: "Block Y", unit: "m" },
  tray_x: { label: "Tray X", unit: "m" },
  tray_y: { label: "Tray Y", unit: "m" },
};

export interface EstimatedCause {
  cause: string;
  confidence: number;
}

export interface SimChange {
  parameter: SimParameter;
  min: number;
  max: number;
  clamped: boolean;
}

export interface FailureDiagnosis {
  success: boolean;
  stage: Stage;
  failure: FailureMode;
  confidence: number;
  estimated_causes: EstimatedCause[];
  recommended_sim_changes: SimChange[];
  summary: string;
}

export interface MetricRow {
  label: string;
  v0: string | null;
  v1: string | null;
}

export interface PolicyMetrics {
  rows: MetricRow[];
  held_out_scenarios: number;
  initial_demos: number;
  corrective_demos: number;
}

export type StageStatus = "passed" | "failed" | "not_reached";

/**
 * Derive the per-stage checklist from the single reported failure stage.
 * Mirrors FailureDiagnosis.stage_timeline() — the critic reports one stage,
 * the checklist is a consequence of it, never a separate field.
 */
export function stageTimeline(d: FailureDiagnosis): Array<[OrderedStage, StageStatus]> {
  if (d.success) return ORDERED_STAGES.map((s) => [s, "passed"]);

  const failedAt = ORDERED_STAGES.indexOf(d.stage as OrderedStage);
  if (failedAt === -1) return ORDERED_STAGES.map((s) => [s, "not_reached"]);

  return ORDERED_STAGES.map((s, i) => [
    s,
    i < failedAt ? "passed" : i === failedAt ? "failed" : "not_reached",
  ]);
}

// --- events (events.py) ------------------------------------------------------

export const LOOP_STAGES = [
  "simulate",
  "verify",
  "deploy",
  "experience",
  "diagnose",
  "adapt",
  "learn",
] as const;
export type LoopStage = (typeof LOOP_STAGES)[number];

interface Envelope {
  ts: number;
  run_id: string;
  seq: number;
}

export type RobotEvent = Envelope &
  (
    | { type: "sim_started"; policy: string; scenario: string; n_episodes: number }
    | { type: "sim_progress"; completed: number; total: number; successes: number }
    | { type: "sim_passed"; successes: number; total: number; threshold: number }
    | { type: "sim_failed"; successes: number; total: number; threshold: number }
    | { type: "deploy_started"; policy: string; condition: string }
    | { type: "video_ready"; url: string; duration_s: number | null; poster_url: string | null }
    | { type: "real_failed"; policy: string; note: string }
    | { type: "real_success"; policy: string; note: string }
    | { type: "critic_started"; video_url: string }
    | { type: "diagnosis_ready"; diagnosis: FailureDiagnosis; latency_ms: number | null }
    | {
        type: "curriculum_ready";
        curriculum_id: string;
        changes: SimChange[];
        n_scenarios: number;
        baseline: Record<string, number>;
        unmappable_reason: string | null;
      }
    | { type: "batch_started"; curriculum_id: string; n_scenarios: number }
    | { type: "batch_progress"; index: number; total: number; success: boolean; thumbnail_url: string | null }
    | { type: "batch_complete"; successes: number; total: number }
    | { type: "policy_loaded"; policy: string; checkpoint: string | null }
    | { type: "redeploy_started"; policy: string; condition: string }
    | { type: "metrics_ready"; metrics: PolicyMetrics }
    | { type: "error"; message: string; where: string }
  );

export type EventType = RobotEvent["type"];

/**
 * Which rail stage lights up for each event. Mirrors LOOP_STAGE_FOR_EVENT.
 *
 * `policy_loaded` is deliberately absent: it fires both before the first
 * deployment and after retraining, so railing it would jump the highlight to
 * LEARN before the robot had run once.
 */
export const LOOP_STAGE_FOR_EVENT: Partial<Record<EventType, LoopStage>> = {
  sim_started: "simulate",
  sim_progress: "simulate",
  sim_passed: "verify",
  sim_failed: "simulate",
  deploy_started: "deploy",
  video_ready: "experience",
  real_failed: "experience",
  real_success: "experience",
  critic_started: "diagnose",
  diagnosis_ready: "diagnose",
  curriculum_ready: "adapt",
  batch_started: "adapt",
  batch_progress: "adapt",
  batch_complete: "adapt",
  metrics_ready: "learn",
  redeploy_started: "deploy",
};

export interface FixtureFile {
  meta: {
    source: string;
    run_id: string;
    description: string;
    warning: string;
    event_count: number;
    duration_s: number;
  };
  events: RobotEvent[];
}
