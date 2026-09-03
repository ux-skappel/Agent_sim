"""Optional LLM-backed minds.

By default the agents think locally (sim/agent.py). Pass --llm and each
agent's choice comes from a real Claude call instead.

The rules of the simulation do not change: the prompt states no task and no
objective, and the model is shown one agent's own perception and memory and
nothing else.

Getting 100 of these into one box is a cost problem, not a code problem.
Three levers live in this file, and they compose:

  1. model choice     -- Haiku is ~15x cheaper per decision than Opus here,
                         mostly because it does not spend output tokens on
                         thinking about a choice between eight verbs.
  2. --llm-wake N     -- consult the model only when something new happened.
                         Cuts call volume several-fold. Not free of bias;
                         see WakePolicy.
  3. --llm-batch      -- the Message Batches API: 50% off, and no rate-limit
                         pressure from 100 simultaneous calls.

Requires:  pip install anthropic   and an ANTHROPIC_API_KEY (or `ant auth login`).
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor

DEFAULT_MODEL = "claude-opus-5"

# What each model needs in its request, and what it costs.
#
#   effort     -- accepts output_config.effort. Sending it to a model that
#                 does not support it is a 400, so this is not cosmetic.
#   thinks     -- reasons before answering by default. Thinking tokens are
#                 billed as output, and for a choice between eight verbs they
#                 dominate the bill. This is the single biggest cost fact here.
#   in_/out    -- USD per million tokens.
#   est_out    -- typical output tokens per decision, thinking included.
MODELS = {
    "claude-opus-5":    {"effort": True,  "thinks": True,
                         "in_": 5.0,  "out": 25.0, "est_out": 260},
    "claude-opus-4-8":  {"effort": True,  "thinks": True,
                         "in_": 5.0,  "out": 25.0, "est_out": 260},
    "claude-fable-5-1": {"effort": True,  "thinks": True,
                         "in_": 10.0, "out": 50.0, "est_out": 320},
    "claude-sonnet-5":  {"effort": True,  "thinks": True,
                         "in_": 2.0,  "out": 10.0, "est_out": 240},
    "claude-haiku-4-5": {"effort": False, "thinks": False,
                         "in_": 1.0,  "out": 5.0,  "est_out": 45},
}
EST_IN = 450          # measured: ~246 system + ~170 per-agent context

# Not cached, deliberately. The shared system prompt measures about 246
# tokens; the minimum cacheable prefix is 512 on Claude Opus 5 and 4096 on
# Claude Haiku 4.5. A cache_control marker on a prefix that short is silently
# ignored -- no error, no saving -- so claiming caching here would be a lie.
# Everything after the system prompt is unique to one agent anyway.
SYSTEM = """You are one entity among others in a closed world. You have a name
and a memory of your own. Nothing outside this message exists for you.

Each moment you choose exactly one action:

  idle      do nothing this moment
  wander    take a step in no particular direction
  approach  take one step toward someone you can see (give their name)
  avoid     take one step away from someone you can see (give their name)
  observe   stay still and take in your surroundings
  speak     say something aloud; anyone nearby will hear it
  address   say something aloud aimed at one person by name
  reflect   turn over something already in your memory

There is no task. There is no objective. There is no correct choice, and
no one is keeping score. Doing nothing is exactly as valid as anything else.

When you speak, you may use any sounds you like, including ones you have
heard others use, ones you invent, and the names of people you remember.
Keep an utterance to at most four short words."""

SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string",
                       "enum": ["idle", "wander", "approach", "avoid",
                                "observe", "speak", "address", "reflect"]},
            "target": {"type": ["string", "null"],
                       "description": "name of another entity, or null"},
            "utterance": {"type": ["string", "null"],
                          "description": "what you say aloud, or null"},
        },
        "required": ["action", "target", "utterance"],
        "additionalProperties": False,
    },
}


def _describe(agent, perception, tick):
    """Everything this agent is entitled to know, and nothing else."""
    lines = ["You are %s. This is moment %d." % (agent.name, tick)]

    seen = perception["visible"]
    if seen:
        near = ", ".join(
            "%s (%d steps away)" % (v["name"], max(abs(v["dx"]), abs(v["dy"])))
            for v in seen[:10])
        lines.append("You can see: " + near + ".")
    else:
        lines.append("You can see no one.")

    recent = agent.memory.recent(10)
    if recent:
        lines.append("\nWhat you remember of the last few moments:")
        for e in recent:
            k = e["kind"]
            if k == "heard":
                lines.append("  %s said \"%s\"%s"
                             % (e["speaker"], " ".join(e["tokens"]),
                                " to you" if e.get("directed") else ""))
            elif k == "spoke":
                lines.append("  you said \"%s\"" % " ".join(e["tokens"]))
            elif k == "saw":
                lines.append("  you saw %s" % ", ".join(e["who"][:5]))
            elif k == "reflected":
                lines.append("  you dwelt on \"%s\"" % e.get("token"))
            elif k == "idle":
                lines.append("  you did nothing")
    else:
        lines.append("\nYou remember nothing yet.")

    lex = sorted(agent.memory.lexicon.items(), key=lambda kv: -kv[1])[:8]
    if lex:
        lines.append("\nSounds you have encountered, and how often: "
                     + ", ".join("%s(%d)" % kv for kv in lex))

    known = sorted(agent.memory.acquaintances.items(),
                   key=lambda kv: -(kv[1]["seen"] + kv[1]["heard"]))[:8]
    if known:
        lines.append("People you have met: "
                     + ", ".join("%s (seen %d, heard speak %d)"
                                 % (n, v["seen"], v["heard"])
                                 for n, v in known))

    lines.append("\nChoose your one action for this moment.")
    return "\n".join(lines)


def request_params(model, effort, agent, perception, tick):
    """The request body, shaped for what this particular model accepts."""
    caps = MODELS.get(model, MODELS[DEFAULT_MODEL])
    params = {
        "model": model,
        "max_tokens": 2000,
        "system": SYSTEM,
        "messages": [{"role": "user",
                      "content": _describe(agent, perception, tick)}],
        "output_config": {"format": SCHEMA},
    }
    if caps["effort"]:
        params["output_config"]["effort"] = effort
    return params


def to_decision(agent, perception, data):
    """Turn the model's answer into an action the world will accept.

    A name the agent cannot currently see is refused here, so the model can
    never act on someone it was not shown."""
    act = data.get("action", "idle")
    visible = {v["name"] for v in perception["visible"]}
    target = data.get("target")
    if target not in visible:
        target = None
    if act in ("approach", "avoid"):
        if target is None:
            return {"action": "wander"}
        return {"action": act, "target": target}
    if act in ("speak", "address"):
        said = (data.get("utterance") or "").strip()
        tokens = said.split()[:4] or ["..."]
        if act == "address" and target is None:
            return {"action": "speak", "tokens": tokens}
        d = {"action": act, "tokens": tokens}
        if act == "address":
            d["target"] = target
        return d
    if act not in ("idle", "wander", "observe", "reflect"):
        return {"action": "idle"}
    return {"action": act}


def parse_message(agent, perception, msg):
    if getattr(msg, "stop_reason", None) == "refusal":
        return {"action": "idle"}
    text = next(b.text for b in msg.content if b.type == "text")
    return to_decision(agent, perception, json.loads(text))


class WakePolicy:
    """When is it worth spending a call on this agent?

    An agent is consulted when something has actually changed for it -- it
    heard something, or someone new came into view -- and otherwise at least
    every `max_gap` moments so it never freezes. In between it falls back to
    its own local choice.

    Be clear about what this costs in fidelity: it is a bias. An agent thinks
    harder in company than alone, which is not something the world would
    otherwise impose. It is off by default for that reason. Turn it on when
    you want a long run you can afford, not when you want a clean result.
    """

    def __init__(self, max_gap=8):
        self.max_gap = max_gap
        self.last = {}        # agent id -> (tick, frozenset of names in view)
        self.woke = 0
        self.slept = 0

    def should_ask(self, agent, perception, tick):
        seen = frozenset(v["name"] for v in perception["visible"])
        prev = self.last.get(agent.id)
        ask = True
        if prev is not None:
            last_tick, last_seen = prev
            heard = any(e["kind"] == "heard" and e["tick"] > last_tick
                        for e in agent.memory.episodes)
            ask = (heard or bool(seen - last_seen)
                   or tick - last_tick >= self.max_gap)
        if ask:
            self.last[agent.id] = (tick, seen)
            self.woke += 1
        else:
            self.slept += 1
        return ask


class _Base:
    def __init__(self, model=DEFAULT_MODEL, effort="low", wake=None,
                 on_error=None, client=None):
        if model not in MODELS:
            raise SystemExit("Unknown --llm-model %r. Known: %s"
                             % (model, ", ".join(sorted(MODELS))))
        if client is None:                 # `client` is injected by the tests
            try:
                import anthropic
            except ImportError:
                raise SystemExit(
                    "The --llm mode needs the Anthropic SDK.\n"
                    "  pip install anthropic\n"
                    "and set ANTHROPIC_API_KEY (or run: ant auth login)")
            client = anthropic.Anthropic(max_retries=5)
        self.client = client
        self.model = model
        self.effort = effort
        self.wake = wake
        self.on_error = on_error
        self.calls = self.errors = 0
        self.input_tokens = self.output_tokens = 0
        self.discount = 1.0

    def _count(self, usage):
        self.calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens

    def _split(self, agents, perceptions, tick):
        """Who gets a call this tick, and who thinks locally."""
        if self.wake is None:
            return list(agents), []
        ask, local = [], []
        for a in agents:
            (ask if self.wake.should_ask(a, perceptions[a.id], tick)
             else local).append(a)
        return ask, local

    def _fallback(self, agent, perceptions, tick, exc=None):
        # A failed call must not become a hidden nudge: fall back to the
        # agent's own local choice, exactly as if --llm were off.
        if exc is not None:
            self.errors += 1
            if self.on_error:
                self.on_error(agent, exc)
        return agent.local_decide(perceptions[agent.id], tick)

    def spend(self):
        caps = MODELS[self.model]
        return self.discount * (self.input_tokens * caps["in_"]
                                + self.output_tokens * caps["out"]) / 1e6

    @staticmethod
    def estimate(model, agents, ticks, wake_gap=0, batch=False):
        """Rough, before you spend anything.

        wake_gap>0 assumes the wake policy skips roughly (1 - 1/gap) of ticks
        for a lone agent; company makes it wake more, so this is a floor."""
        caps = MODELS.get(model, MODELS[DEFAULT_MODEL])
        calls = agents * ticks
        if wake_gap:
            calls = int(calls * min(1.0, 1.0 / wake_gap + 0.25))
        cost = calls * (EST_IN * caps["in_"] + caps["est_out"] * caps["out"]) / 1e6
        if batch:
            cost *= 0.5
        return calls, cost


class LiveMinds(_Base):
    """One call per agent per tick, issued in parallel. Immediate, full price."""

    def __init__(self, workers=16, **kw):
        super().__init__(**kw)
        self.workers = workers

    def _one(self, agent, perceptions, tick):
        try:
            msg = self.client.messages.create(
                **request_params(self.model, self.effort, agent,
                                 perceptions[agent.id], tick))
            self._count(msg.usage)
            return parse_message(agent, perceptions[agent.id], msg)
        except Exception as exc:                      # noqa: BLE001
            return self._fallback(agent, perceptions, tick, exc)

    def __call__(self, agents, perceptions, tick):
        ask, local = self._split(agents, perceptions, tick)
        out = [(a, self._fallback(a, perceptions, tick)) for a in local]
        if ask:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = [pool.submit(self._one, a, perceptions, tick)
                           for a in ask]
                out += [(a, f.result()) for a, f in zip(ask, futures)]
        order = {a.id: i for i, a in enumerate(agents)}
        return sorted(out, key=lambda pair: order[pair[0].id])


class BatchMinds(_Base):
    """One batch per tick through the Message Batches API.

    Half price and no rate-limit pressure, at the cost of waiting for the
    batch to finish. That trade is usually right for a simulation: nobody is
    watching a tick happen live, and a long run left overnight is exactly the
    shape of work batches are for."""

    def __init__(self, poll=20.0, timeout=3600.0, on_wait=None, **kw):
        super().__init__(**kw)
        self.poll = poll
        self.timeout = timeout
        self.on_wait = on_wait
        self.discount = 0.5
        self.batches = 0

    def _build(self, ask, perceptions, tick):
        from anthropic.types.message_create_params import (
            MessageCreateParamsNonStreaming)
        from anthropic.types.messages.batch_create_params import Request
        return [Request(custom_id="agent-%d" % a.id,
                        params=MessageCreateParamsNonStreaming(
                            **request_params(self.model, self.effort, a,
                                             perceptions[a.id], tick)))
                for a in ask]

    def _await(self, batch_id):
        deadline = time.time() + self.timeout
        while True:
            batch = self.client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                return batch
            if time.time() > deadline:
                raise TimeoutError("batch %s still %s after %.0fs"
                                   % (batch_id, batch.processing_status,
                                      self.timeout))
            if self.on_wait:
                self.on_wait(batch)
            time.sleep(self.poll)

    def __call__(self, agents, perceptions, tick):
        ask, local = self._split(agents, perceptions, tick)
        decisions = {a.id: self._fallback(a, perceptions, tick) for a in local}
        by_id = {a.id: a for a in agents}

        if ask:
            try:
                created = self.client.messages.batches.create(
                    requests=self._build(ask, perceptions, tick))
                self.batches += 1
                self._await(created.id)
                for res in self.client.messages.batches.results(created.id):
                    agent = by_id[int(res.custom_id.split("-")[1])]
                    if res.result.type == "succeeded":
                        msg = res.result.message
                        self._count(msg.usage)
                        decisions[agent.id] = parse_message(
                            agent, perceptions[agent.id], msg)
                    else:
                        decisions[agent.id] = self._fallback(
                            agent, perceptions, tick,
                            RuntimeError(res.result.type))
            except Exception as exc:                  # noqa: BLE001
                for a in ask:
                    decisions.setdefault(
                        a.id, self._fallback(a, perceptions, tick, exc))

        for a in ask:
            decisions.setdefault(a.id, self._fallback(a, perceptions, tick))
        return [(a, decisions[a.id]) for a in agents]


# Kept for the old name used before batching existed.
LLMMinds = LiveMinds
