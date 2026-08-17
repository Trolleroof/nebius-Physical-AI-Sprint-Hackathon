"use client";

/**
 * The always-visible loop across the top.
 *
 * This is the single most important element on screen: it is the only thing
 * that tells a judge who arrived forty seconds late where they are in the
 * story. It never scrolls away and never empties.
 */

import { motion } from "motion/react";
import { LOOP_STAGES, LoopStage } from "@/lib/contract";
import { RailStatus } from "@/lib/reducer";

const TONE: Record<RailStatus, string> = {
  pending: "text-[var(--color-dim)]",
  active: "text-[var(--color-ink)] glow-accent",
  done: "text-[var(--color-muted)]",
  failed: "text-[var(--color-fail)]",
};

const MARK: Record<RailStatus, string> = {
  pending: "·",
  active: "◆",
  done: "✓",
  failed: "✗",
};

export function LoopRail({
  rail,
  active,
  cycle,
}: {
  rail: Record<LoopStage, RailStatus>;
  active: LoopStage | null;
  cycle: number;
}) {
  return (
    <div className="flex items-center justify-between gap-1 px-2">
      {LOOP_STAGES.map((stage, i) => {
        const status = rail[stage];
        const isActive = stage === active;
        return (
          <div key={stage} className="flex flex-1 items-center gap-1">
            <motion.div
              className="relative flex flex-1 flex-col items-center gap-1 py-1"
              animate={{ scale: isActive ? 1.04 : 1 }}
              transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
            >
              <span className={`text-[10px] leading-none ${TONE[status]}`}>{MARK[status]}</span>
              <span
                className={`font-mono text-[10px] tracking-[0.2em] uppercase transition-colors duration-500 ${TONE[status]}`}
              >
                {stage}
              </span>
              {isActive && (
                <motion.span
                  layoutId="rail-underline"
                  className="absolute -bottom-0.5 h-px w-8 bg-[var(--color-accent)]"
                  transition={{ type: "spring", stiffness: 260, damping: 30 }}
                />
              )}
            </motion.div>

            {i < LOOP_STAGES.length - 1 && (
              <span className="h-px w-3 shrink-0 bg-[var(--color-line)]" />
            )}
          </div>
        );
      })}

      {/* Loop-back marker: the arrow that makes it a cycle rather than a pipeline. */}
      <div className="flex shrink-0 items-center gap-1 pl-1">
        <span className="h-px w-3 bg-[var(--color-line)]" />
        <span className="font-mono text-[11px] text-[var(--color-accent)]">↺</span>
        {cycle > 0 && (
          <span className="label text-[var(--color-accent)]">lap {cycle + 1}</span>
        )}
      </div>
    </div>
  );
}
