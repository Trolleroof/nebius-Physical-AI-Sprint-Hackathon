"use client";

/**
 * The dashboard. One screen, fixed to the viewport, never scrolls.
 *
 * Layout mirrors the architecture: evidence on the left (what the robot did),
 * reasoning on the right (what the system concluded), and the payoff pinned
 * across the bottom where it stays visible whatever else is happening.
 */

import { BatchGrid } from "@/components/BatchGrid";
import { Header, LeftRail, RightRail } from "@/components/Chrome";
import { DemoControls } from "@/components/DemoControls";
import { FailurePanel } from "@/components/FailurePanel";
import { ImprovementStrip } from "@/components/ImprovementStrip";
import { LoopRail } from "@/components/LoopRail";
import { RealityToSim } from "@/components/RealityToSim";
import { StageView } from "@/components/StageView";
import { useRun } from "@/lib/useRun";

export default function Dashboard() {
  const replay = useRun();
  const { state } = replay;

  return (
    <div className="relative flex h-full flex-col">
      <div className="ambient" />

      <div className="relative z-10 flex h-full flex-col">
        <Header state={state} source={replay.source} />

        <div className="shrink-0 border-b border-[var(--color-line)] py-2">
          <LoopRail rail={state.rail} active={state.activeStage} cycle={state.cycle} />
        </div>

        <div className="flex min-h-0 flex-1">
          <LeftRail state={state} />

          <main className="grid min-h-0 flex-1 grid-cols-1 grid-rows-[1.55fr_1fr] gap-2 p-2 lg:grid-cols-[1.6fr_1fr]">
            {/* Evidence column */}
            <StageView state={state} />
            {/* Reasoning column */}
            <FailurePanel state={state} />
            <BatchGrid state={state} />
            <RealityToSim state={state} />
          </main>

          <RightRail />
        </div>

        <div className="shrink-0 px-2 pb-2">
          <ImprovementStrip state={state} />
        </div>
      </div>

      <DemoControls replay={replay} />
    </div>
  );
}
