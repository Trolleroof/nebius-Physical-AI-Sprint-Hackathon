"use client";

/**
 * Per-panel data provenance.
 *
 * Three of the four subsystems are usually standing in, and which three
 * changes through the day. Rather than trusting whoever is presenting to
 * remember which is which, every panel carries a label taken from the
 * backend itself (plan section 20).
 *
 * Replay overrides everything: if the dashboard is playing a recording, no
 * panel may claim to be live, however real the subsystem was when the
 * recording was made.
 */

import { useEffect, useState } from "react";
import { Source as RunSource } from "./useRun";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const REFRESH_MS = 5000;

export type Mode = "live" | "mock" | "scripted" | "replay";
export type Subsystem = "critic" | "sim" | "robot" | "metrics";

export interface Origin {
  mode: Mode;
  detail: string;
}

export type Provenance = Record<Subsystem, Origin>;

const REPLAYED: Origin = { mode: "replay", detail: "recorded run" };
const UNKNOWN: Origin = { mode: "mock", detail: "backend not connected" };

export function useProvenance(runSource: RunSource): Provenance {
  const [live, setLive] = useState<Provenance | null>(null);

  useEffect(() => {
    if (runSource !== "live") return;

    let cancelled = false;
    const load = () =>
      fetch(`${API}/api/provenance`)
        .then((r) => r.json())
        .then((p: Provenance) => !cancelled && setLive(p))
        .catch(() => {});

    load();
    // Provenance changes mid-run — metrics flip to measured the moment
    // someone POSTs real figures — so this is polled rather than fetched once.
    const timer = setInterval(load, REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [runSource]);

  if (runSource !== "live") {
    return { critic: REPLAYED, sim: REPLAYED, robot: REPLAYED, metrics: REPLAYED };
  }
  return live ?? { critic: UNKNOWN, sim: UNKNOWN, robot: UNKNOWN, metrics: UNKNOWN };
}
