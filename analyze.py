#!/usr/bin/env python3
"""Look for things that emerged.

These measurements are for the human observer. The agents never see any of
them, and nothing in the simulation is tuned to make these numbers go up.

    python3 analyze.py runs/seed1     # re-analyse a finished run
"""

import json
import os
import random
import sys

from sim.lexicon import SEED_WORDS


# -- clustering ----------------------------------------------------------
def components(positions, radius):
    """Connected groups: agents linked if within `radius` of each other."""
    n = len(positions)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        xi, yi = positions[i]
        for j in range(i + 1, n):
            xj, yj = positions[j]
            if abs(xi - xj) <= radius and abs(yi - yj) <= radius:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=len, reverse=True)


def random_baseline(n, w, h, radius, trials=12, seed=0):
    """What clustering looks like with no agents at all -- just n dots
    scattered at random. Anything above this is the interesting part."""
    rng = random.Random(seed)
    sizes = []
    for _ in range(trials):
        pos = [(rng.randrange(w), rng.randrange(h)) for _ in range(n)]
        sizes.append(len(components(pos, radius)[0]))
    return sum(sizes) / len(sizes)


def _bar(frac, width=28):
    filled = int(round(frac * width))
    return "#" * filled + "." * (width - filled)


def report(world):
    n = len(world.agents)
    lines = []
    add = lines.append
    add("=" * 62)
    add("WHAT EMERGED  (tick %d, %d agents, no goals given)" % (world.tick_no, n))
    add("=" * 62)

    # --- activity -------------------------------------------------------
    total = sum(world.action_counts.values()) or 1
    add("")
    add("-- what they spent their time doing --")
    for act, c in sorted(world.action_counts.items(), key=lambda kv: -kv[1]):
        add("  %-9s %6d  %5.1f%%  %s" % (act, c, 100.0 * c / total,
                                         _bar(c / total)))
    doing_nothing = (world.action_counts.get("idle", 0)
                     + world.action_counts.get("reflect", 0)) / total
    add("  inactivity (idle + reflect): %.1f%%" % (100 * doing_nothing))

    # --- space ----------------------------------------------------------
    pos = [(a.x, a.y) for a in world.agents]
    groups = components(pos, world.vision)
    base = random_baseline(n, world.width, world.height, world.vision)
    add("")
    add("-- groups in space (linked if within vision, %d cells) --" % world.vision)
    add("  clusters: %d   largest: %d agents   random baseline largest: %.1f"
        % (len(groups), len(groups[0]), base))
    sizes = [len(g) for g in groups if len(g) > 1][:8]
    add("  cluster sizes: %s" % (sizes or "none larger than 1"))
    alone = sum(1 for g in groups if len(g) == 1)
    add("  agents standing alone: %d" % alone)

    # --- shared vocabulary ---------------------------------------------
    doms = {}
    for a in world.agents:
        d = a.memory.dominant_token()
        doms[d] = doms.get(d, 0) + 1
    top = sorted(doms.items(), key=lambda kv: -kv[1])[:6]
    unit = "sound" if world.speech_mode == "tokens" else "word"
    add("")
    add("-- shared vocabulary (most-repeated %s in each private memory) --"
        % unit)
    add("  distinct %ss ever uttered: %d" % (unit, len(world.token_ids)))
    add("  distinct 'favourite' %ss across the population: %d"
        % (unit, len(doms)))
    if world.speech_mode == "words":
        add("  NOTE: in --speech words the pool is a fixed list of %d supplied"
            % len(SEED_WORDS))
        add("        words, so agreement is partly the small pool, not only")
        add("        contact. Compare against a --speech tokens run.")
    for tok, c in top:
        add("  %-10s held by %3d agents  %s"
            % (tok if tok else "(silence)", c, _bar(c / n)))

    # --- who gets attention --------------------------------------------
    heard = {a.name: 0 for a in world.agents}
    spoke_at = {a.name: 0 for a in world.agents}
    knows = {a.name: 0 for a in world.agents}
    for a in world.agents:
        for other, imp in a.memory.acquaintances.items():
            if other in heard:
                heard[other] += imp["heard"]
                spoke_at[other] += imp["addressed_me"]
        knows[a.name] = len(a.memory.acquaintances)
    got = world.addressed_counts
    add("")
    add("-- who ended up at the centre of attention --")
    add("  heard      = times someone else was in earshot of them")
    add("  addressed  = times another agent aimed speech at them by name")
    for name, c in sorted(heard.items(), key=lambda kv: -kv[1])[:8]:
        add("  %-10s heard %5d   addressed %4d   spoke-at-others %4d"
            "   knows %3d"
            % (name, c, got.get(name, 0), spoke_at[name], knows[name]))
    if got:
        star = max(got.items(), key=lambda kv: kv[1])
        add("  most spoken-to: %s (%d times)" % star)
    add("  most isolated: %s"
        % ", ".join("%s(%d)" % (nm, k) for nm, k in
                    sorted(knows.items(), key=lambda kv: kv[1])[:5]))

    # --- how connected the population is -------------------------------
    avg_known = sum(knows.values()) / n
    add("")
    add("-- acquaintance network --")
    add("  average number of others an agent has met: %.1f of %d"
        % (avg_known, n - 1))
    add("  utterances spoken in total: %d" % world.utterance_count)
    add("")
    add("Read runs/<dir>/conversations.log for the transcript,")
    add("events.jsonl for every action, agents.json for individual minds.")
    return "\n".join(lines)


def report_from_dir(run_dir):
    """Lighter re-analysis from the saved frames (no live world needed)."""
    with open(os.path.join(run_dir, "frames.json"), encoding="utf-8") as f:
        data = json.load(f)
    meta, frames = data["meta"], data["frames"]
    last = frames[-1]
    pos = [(a[0], a[1]) for a in last["a"]]
    groups = components(pos, meta["vision"])
    base = random_baseline(len(pos), meta["width"], meta["height"],
                           meta["vision"])
    doms = {}
    for a in last["a"]:
        doms[a[2]] = doms.get(a[2], 0) + 1
    tokens = meta["tokens"]
    out = ["run: %s" % run_dir,
           "ticks: %d   agents: %d" % (meta["ticks"], len(pos)),
           "clusters: %d  largest: %d  (random baseline %.1f)"
           % (len(groups), len(groups[0]), base),
           "distinct favourite sounds: %d" % len(doms)]
    for tid, c in sorted(doms.items(), key=lambda kv: -kv[1])[:5]:
        name = tokens[tid] if 0 <= tid < len(tokens) else "(silence)"
        out.append("  %-10s %d agents" % (name, c))
    return "\n".join(out)


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "runs/seed1"
    print(report_from_dir(d))
