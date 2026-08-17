"use client";

/**
 * Targeted Antioch runs, one cell per scenario.
 *
 * This is the flywheel made visible: one real failure filling a grid with
 * corrective experience. Cells land one at a time so the grid visibly fills
 * during the demo rather than appearing complete.
 */

import { motion } from "motion/react";
import { DashboardState } from "@/lib/reducer";
import { Awaiting, Panel } from "./ui";

export function BatchGrid({ state }: { state: DashboardState }) {
  const batch = state.batch;
  const done = batch ? batch.results.filter((r) => r !== null).length : 0;
  const passed = batch ? batch.results.filter((r) => r === true).length : 0;

  return (
    <Panel
      title="Targeted Antioch runs"
      live={!!batch && !batch.complete}
      right={
        batch && (
          <span className="font-mono text-[10px] text-[var(--color-muted)]">
            {done}/{batch.total}
            {batch.complete && (
              <span className="ml-2 text-[var(--color-ok)]">{passed} passed</span>
            )}
          </span>
        )
      }
    >
      {!batch ? (
        <Awaiting label="awaiting curriculum" />
      ) : (
        <div className="grid h-full grid-cols-10 content-start gap-1">
          {batch.results.map((result, i) => (
            <motion.div
              key={i}
              initial={false}
              animate={{
                opacity: result === null ? 0.25 : 1,
                scale: result === null ? 0.9 : 1,
              }}
              transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
              className={`flex aspect-square items-center justify-center rounded-[1px] border text-[8px] ${
                result === null
                  ? "border-[var(--color-line)] text-transparent"
                  : result
                    ? "border-[var(--color-ok)]/40 bg-[var(--color-ok)]/10 text-[var(--color-ok)]"
                    : "border-[var(--color-fail)]/40 bg-[var(--color-fail)]/10 text-[var(--color-fail)]"
              }`}
            >
              {result === null ? "" : result ? "✓" : "✗"}
            </motion.div>
          ))}
        </div>
      )}
    </Panel>
  );
}
