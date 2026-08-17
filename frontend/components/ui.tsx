"use client";

import { motion } from "motion/react";
import { ReactNode } from "react";

/**
 * Shared HUD primitives. Panels never appear or disappear during a run —
 * they populate. An empty panel keeps its frame and shows placeholder rows so
 * the layout never reflows, which is the difference between reading as a
 * product and reading as a script printing to a terminal.
 */

export function Panel({
  title,
  right,
  children,
  className = "",
  live = false,
}: {
  title: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  live?: boolean;
}) {
  return (
    <section className={`panel bracket flex min-h-0 flex-col ${className}`}>
      <header className="flex shrink-0 items-center gap-3 border-b border-[var(--color-line)] px-3 py-2">
        <h2 className="label text-[var(--color-ink)]">{title}</h2>
        <div className="hatch h-2 flex-1 opacity-30" />
        {live && <span className="size-1.5 rounded-full bg-[var(--color-accent)] pulse" />}
        {right}
      </header>
      <div className="min-h-0 flex-1 overflow-hidden p-3">{children}</div>
    </section>
  );
}

/** Placeholder shown before a panel has any data. */
export function Awaiting({ label = "awaiting data" }: { label?: string }) {
  return (
    <div className="flex h-full items-center justify-center">
      <span className="label text-[var(--color-dim)]">{label}</span>
    </div>
  );
}

/** A number that counts up when it changes, so metrics feel measured. */
export function Ticker({ value, className = "" }: { value: string; className?: string }) {
  return (
    <motion.span
      key={value}
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
      className={className}
    >
      {value}
    </motion.span>
  );
}

/** Horizontal confidence / proportion bar. */
export function Meter({ value, tone = "accent" }: { value: number; tone?: "accent" | "dim" }) {
  const colour = tone === "accent" ? "var(--color-accent)" : "var(--color-dim)";
  return (
    <div className="h-1 w-full overflow-hidden rounded-full bg-[var(--color-line)]">
      <motion.div
        className="h-full rounded-full"
        style={{ background: colour }}
        initial={{ width: 0 }}
        animate={{ width: `${Math.round(value * 100)}%` }}
        transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
      />
    </div>
  );
}

export function StatusDot({ tone }: { tone: "ok" | "fail" | "run" | "idle" }) {
  const map = {
    ok: "bg-[var(--color-ok)]",
    fail: "bg-[var(--color-fail)]",
    run: "bg-[var(--color-accent)] pulse",
    idle: "bg-[var(--color-dim)]",
  } as const;
  return <span className={`inline-block size-1.5 rounded-full ${map[tone]}`} />;
}
