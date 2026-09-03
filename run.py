#!/usr/bin/env python3
"""Run the simulation.

    python3 run.py                      # 100 agents, 600 ticks
    python3 run.py --ticks 300 --live   # watch it in the terminal
    python3 run.py --seed 7 --out runs/seed7

Nothing here tells the agents what to do.
"""

import argparse
import json
import os
import sys
import time

from sim.world import World
from sim.recorder import Recorder
import analyze
import viz

PALETTE = "abcdefghijklmnopqrstuvwxyz"


def render_terminal(world, cols=64, rows=32):
    """A coarse ASCII picture of the world. Letters group agents that
    currently share a most-frequent token, so shared vocabulary shows up as
    patches of the same letter."""
    sx = max(1, world.width // cols + (world.width % cols > 0))
    sy = max(1, world.height // rows + (world.height % rows > 0))
    grid = [[" "] * ((world.width + sx - 1) // sx)
            for _ in range((world.height + sy - 1) // sy)]
    for a in world.agents:
        dom = a.memory.dominant_token()
        ch = PALETTE[world.token_id(dom) % 26] if dom else "."
        gy, gx = a.y // sy, a.x // sx
        cur = grid[gy][gx]
        grid[gy][gx] = ch if cur == " " else (cur.upper() if cur.islower() else cur)
    out = ["+" + "-" * len(grid[0]) + "+"]
    out += ["|" + "".join(r) + "|" for r in grid]
    out.append("+" + "-" * len(grid[0]) + "+")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description="Goal-free multi-agent simulation")
    p.add_argument("--agents", type=int, default=100)
    p.add_argument("--ticks", type=int, default=600)
    p.add_argument("--width", type=int, default=90)
    p.add_argument("--height", type=int, default=90)
    p.add_argument("--vision", type=int, default=6)
    p.add_argument("--hearing", type=int, default=6)
    p.add_argument("--memory", type=int, default=500,
                   help="episodes kept per agent")
    p.add_argument("--speech", default="tokens", choices=["tokens", "words"],
                   help="tokens: invented syllables, cleanly measurable. "
                        "words: real language, far more human, but the word "
                        "pool is supplied rather than invented")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default=None, help="run directory (default runs/<seed>)")
    p.add_argument("--live", action="store_true", help="ASCII view while running")
    p.add_argument("--fps", type=float, default=12.0, help="live view speed")
    p.add_argument("--no-viz", action="store_true")
    p.add_argument("--llm", action="store_true",
                   help="let a real Claude call make each agent's choice "
                        "(costs money; see README)")
    p.add_argument("--llm-model", default="claude-opus-5")
    p.add_argument("--llm-effort", default="low",
                   choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--llm-batch", action="store_true",
                   help="use the Message Batches API: half price, but each "
                        "tick waits for the batch to finish")
    p.add_argument("--llm-wake", type=int, default=0, metavar="N",
                   help="only consult the model when something changed for "
                        "an agent, and at least every N moments. Cheaper, "
                        "but biases agents toward thinking harder in company")
    p.add_argument("--yes", action="store_true",
                   help="skip the cost confirmation for --llm")
    args = p.parse_args(argv)

    minds = None
    if args.llm:
        from sim import llm_mind
        calls, cost = llm_mind._Base.estimate(
            args.llm_model, args.agents, args.ticks,
            wake_gap=args.llm_wake, batch=args.llm_batch)
        print("--llm: %s Claude calls on %s%s%s"
              % ("{:,}".format(calls), args.llm_model,
                 " (batched, half price)" if args.llm_batch else "",
                 " (waking every <=%d moments)" % args.llm_wake
                 if args.llm_wake else ""))
        print("       rough cost: about $%.2f" % cost)
        if not args.llm_batch and args.llm_model != "claude-haiku-4-5":
            cheap, ccost = llm_mind._Base.estimate(
                "claude-haiku-4-5", args.agents, args.ticks,
                wake_gap=args.llm_wake or 8, batch=True)
            print("       for comparison: --llm-model claude-haiku-4-5 "
                  "--llm-batch --llm-wake 8 is about $%.2f" % ccost)
        if not args.yes:
            if input("       continue? [y/N] ").strip().lower() != "y":
                return None
        wake = llm_mind.WakePolicy(args.llm_wake) if args.llm_wake else None
        cls = llm_mind.BatchMinds if args.llm_batch else llm_mind.LiveMinds
        kw = {"model": args.llm_model, "effort": args.llm_effort, "wake": wake}
        if args.llm_batch:
            kw["on_wait"] = lambda b: print("       batch %s: %d done"
                                            % (b.processing_status,
                                               b.request_counts.succeeded))
        minds = cls(**kw)

    run_dir = args.out or os.path.join("runs", "seed%d" % args.seed)
    rec = Recorder(run_dir)
    world = World(width=args.width, height=args.height, n_agents=args.agents,
                  vision=args.vision, hearing=args.hearing, seed=args.seed,
                  memory_capacity=args.memory, decider=minds,
                  speech_mode=args.speech)

    print("world %dx%d  agents=%d  vision=%d  ticks=%d  seed=%d  speech=%s"
          % (args.width, args.height, args.agents, args.vision,
             args.ticks, args.seed, args.speech))
    print("logging to %s/" % run_dir)

    t0 = time.time()
    for i in range(args.ticks):
        world.step(rec)
        if args.live:
            sys.stdout.write("\033[H\033[J")
            sys.stdout.write("tick %d/%d   distinct sounds: %d\n"
                             % (world.tick_no, args.ticks, len(world.token_ids)))
            sys.stdout.write(render_terminal(world) + "\n")
            sys.stdout.flush()
            time.sleep(1.0 / max(0.1, args.fps))
        elif i % 50 == 0:
            print("  tick %d/%d" % (i, args.ticks))

    meta = {"width": args.width, "height": args.height, "ticks": args.ticks,
            "seed": args.seed, "vision": args.vision, "hearing": args.hearing,
            "speech": args.speech,
            "names": [a.name for a in world.agents],
            "tokens": [t for t, _ in sorted(world.token_ids.items(),
                                            key=lambda kv: kv[1])]}
    rec.close(meta)

    # Each agent's final private state, so you can read individual minds.
    with open(os.path.join(run_dir, "agents.json"), "w", encoding="utf-8") as f:
        json.dump([{"id": a.id, "name": a.name, "pos": [a.x, a.y],
                    "temperament": {k: round(v, 4) for k, v in
                                    a.temperament.items()},
                    "invent_rate": round(a.invent_rate, 3),
                    "memory": a.memory.snapshot()}
                   for a in world.agents], f, indent=1)

    if minds is not None:
        print("\nClaude calls: %d   failed: %d   spent: $%.2f"
              % (minds.calls, minds.errors, minds.spend()))
        if minds.wake is not None:
            total = minds.wake.woke + minds.wake.slept
            print("Woken: %d of %d possible calls (%.0f%% saved)"
                  % (minds.wake.woke, total,
                     100.0 * minds.wake.slept / max(1, total)))

    report = analyze.report(world)
    with open(os.path.join(run_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)

    if not args.no_viz:
        html = viz.build(run_dir)
        print("visualisation: %s" % html)
    print("done in %.1fs" % (time.time() - t0))
    return run_dir


if __name__ == "__main__":
    main()
