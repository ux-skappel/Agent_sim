# Agent_sim — 100 agents, a closed world, no purpose

100 agents are placed in a bounded grid. They can look around, move, speak,
address each other by name, remember, and do nothing. **No agent is given a
goal, a role, a reward, or a hidden incentive.** Then we watch.

Pure Python 3 standard library. Nothing to install.

## Run it

```bash
python3 run.py                  # 100 agents, 600 ticks, ~4 seconds
python3 run.py --ticks 300 --live   # watch it in the terminal
python3 run.py --seed 7             # a different world
```

Then open the replay in a browser:

```
runs/seed1/replay.html
```

One self-contained file — press **Play**. Each dot is an agent. Its colour is
the sound it has heard or said most often, which is the closest thing it has to
a belief. Rings are agents speaking; lines are agents speaking to someone by
name.

## What you get in `runs/<name>/`

| file | what's in it |
|---|---|
| `replay.html` | the visualisation — open it in a browser |
| `conversations.log` | every utterance, who said it, who was in earshot |
| `events.jsonl` | every action by every agent on every tick |
| `agents.json` | each agent's final private memory and temperament |
| `report.txt` | the emergence summary printed at the end |
| `frames.json` | raw positions per tick (feeds the replay) |

## How an agent works

Each agent has:

* **An identity** — a unique name and a temperament drawn at birth.
* **A private memory** — episodes it lived through, a lexicon of sounds it has
  heard, and an impression of every agent it has personally met. Nothing is
  shared. An agent that has never met you does not know you exist.
* **Eight things it can do** on any tick, and no reason to prefer any of them:

  `idle` · `wander` · `approach someone` · `avoid someone` · `observe` ·
  `speak` · `address someone by name` · `reflect`

Every tick: everyone perceives the world as it stands, everyone decides
privately and simultaneously, actions take effect, and speech is delivered to
whoever happened to be in earshot.

### How the "no hidden goals" rule is actually enforced

This is the part worth being strict about, so:

* **No utility, reward, fitness or score exists anywhere in the code.** Nothing
  is maximised. Grep for it.
* **Every social action has an equally weighted opposite.** `approach` and
  `avoid` are drawn from the same distribution and pick their target using the
  *same* salience number. Nothing tilts the world toward togetherness.
* **Temperament is symmetric noise.** Each agent's eight action weights are
  eight independent draws from one identical distribution, then normalised.
  That makes agents *different from each other*; it does not make any of them
  *want* anything.
* **Memory drives content, not direction.** Memory decides *what* an agent says
  and *who* comes to mind — never whether saying it is a good idea.
* **The environment is inert.** No food, no hazards, no resources, no
  scoreboard. Time passes; that is all it offers.

The statistics in `report.txt` are computed for you, the observer. No agent can
read them, and nothing in the simulation is tuned to make them go up.

## Where speech comes from

An agent utters one to three sounds. Each is either copied from its own lexicon
(weighted by how often it has encountered it) or freshly invented. Sometimes it
says a name it remembers. Copying is frequency-weighted with no feedback rule —
so if the population converges on a shared word, that is drift and social
contact doing it, not a rule telling them to agree.

## Reading the results

`report.txt` measures four things, each against an honest baseline:

* **Time use** — including how much of the population simply does nothing.
* **Groups in space** — clusters compared against the same number of dots
  scattered at random. *Above* the baseline means real gathering. *Below* it
  means the agents are actively spreading out.
* **Shared vocabulary** — how many distinct "favourite" sounds survive in 100
  private memories. Far fewer than were invented means a common tongue formed.
* **Attention** — who was heard most, who got addressed by name most, and who
  never met anybody.

Re-analyse a finished run without re-running it:

```bash
python3 analyze.py runs/seed1
```

## Optional: real Claude minds, and how to afford 100 of them

By default the agents think locally — the choosing is stochastic Python, not a
language model. That is what makes 100 agents × 600 ticks finish in four
seconds.

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...        # or: ant auth login

python3 run.py --llm --agents 12 --ticks 40      # start here
```

`run.py` prints an estimate and asks before spending anything.

### The cost problem, stated plainly

One call per agent per tick means 100 × 600 = **60,000 calls** for a default
run. Naively that is a few hundred dollars. Three levers change that, and they
multiply:

| Configuration | Calls | Cost |
|---|---:|---:|
| Opus 5, every agent every tick | 60,000 | ~$525 |
| Opus 5 + `--llm-batch` | 60,000 | ~$263 |
| Opus 5 + batch + `--llm-wake 8` | 22,500 | ~$98 |
| Sonnet 5 + batch + wake | 22,500 | ~$37 |
| **Haiku 4.5 + batch + wake** | **22,500** | **~$8** |

```bash
python3 run.py --llm --llm-model claude-haiku-4-5 --llm-batch --llm-wake 8
```

**1. The model matters more than anything else.** Not because Haiku is cheaper
per token — because it does not spend output tokens *thinking* before choosing
between eight verbs. On Opus 5 the reasoning tokens, billed as output,
dominate the bill for a decision this small. That single fact is most of the
15× gap.

**2. `--llm-batch` — the Message Batches API.** A tick is 100 independent
requests, which is exactly the shape batches are for: half price, and no
rate-limit pressure from 100 simultaneous calls. The cost is waiting for each
batch to finish, so a tick takes minutes instead of seconds. For a simulation
nobody watches live, that is the right trade — start a long run and come back.

**3. `--llm-wake N` — only think when something happened.** An agent is
consulted when it heard something or someone new came into view, and otherwise
at least every N moments so it never freezes. In between it uses its local
choice.

> **Be careful with this one.** It is *not* neutral. An agent thinks harder in
> company than alone, which is a bias the world would not otherwise impose. It
> is off by default. Turn it on when you want a long run you can afford — not
> when you want a clean result about whether groups form.

### What does *not* help

Prompt caching. The shared system prompt measures ~246 tokens; the minimum
cacheable prefix is 512 on Opus 5 and 4096 on Haiku 4.5. A cache marker on a
prefix that short is silently ignored — no error, no saving. Everything after
the system prompt is unique to one agent anyway. There is deliberately no
`cache_control` in `sim/llm_mind.py`, and a test asserts its absence so nobody
adds one back believing it does something.

### What stays true in LLM mode

The prompt in `sim/llm_mind.py` states no task, no objective and no preferred
behaviour. The model sees one agent's own perception and memory and nothing
else — a test checks the prompt never names an agent the subject has not met.
A name the agent cannot currently see is refused before it reaches the world.
And if a call fails, that agent falls back to its local choice, so a network
error never becomes a hidden nudge. Read that prompt before you trust any
result from this mode.

## Knobs

```
--agents 100     --ticks 600      --seed 1
--width 90       --height 90      # world size (walls, closed)
--vision 6       --hearing 6      # how far an agent can see / be heard
--memory 300     # episodes kept per agent before the oldest is forgotten
--live --fps 12  # terminal view
--no-viz         # skip building replay.html

--llm                            # use real Claude calls (see above)
--llm-model claude-haiku-4-5     # default claude-opus-5
--llm-effort low                 # low | medium | high | xhigh | max
--llm-batch                      # Message Batches API: half price, slower
--llm-wake 8                     # only think when something changed
--yes                            # skip the cost confirmation
```

Density is the interesting dial. A smaller world makes encounters constant; a
larger one makes them rare, and rare encounters are how you find out whether
anything holds together on its own.

## Tests

```bash
python3 test_sim.py
```

These check the parts that matter: same seed gives the same world, agents only
know what they personally perceived, and no reward signal has crept in.
