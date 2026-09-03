"""An optional LLM-backed mind.

By default the agents in this simulation think locally (see sim/agent.py).
Pass --llm to run.py and each agent's choice is made by a real Claude call
instead: one call per agent per tick.

The rules of the simulation do not change. The prompt below deliberately
states no task, no objective and no preferred behaviour, and the model is
shown nothing except that one agent's own perception and memory.

Requires:  pip install anthropic   and an ANTHROPIC_API_KEY (or `ant auth login`).
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

DEFAULT_MODEL = "claude-opus-5"

# Deliberately goal-free. Nothing here suggests what is worth doing.
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


class LLMMinds:
    """Decides for every agent in a tick, in parallel."""

    def __init__(self, model=DEFAULT_MODEL, effort="low", workers=16,
                 on_error=None):
        try:
            import anthropic
        except ImportError:
            raise SystemExit(
                "The --llm mode needs the Anthropic SDK.\n"
                "  pip install anthropic\n"
                "and set ANTHROPIC_API_KEY (or run: ant auth login)")
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.workers = workers
        self.on_error = on_error
        self.calls = 0
        self.errors = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def _one(self, agent, perception, tick):
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=[{"type": "text", "text": SYSTEM,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user",
                           "content": _describe(agent, perception, tick)}],
                output_config={"effort": self.effort, "format": SCHEMA},
            )
            self.calls += 1
            self.input_tokens += resp.usage.input_tokens
            self.output_tokens += resp.usage.output_tokens
            if resp.stop_reason == "refusal":
                return {"action": "idle"}
            text = next(b.text for b in resp.content if b.type == "text")
            return self._to_decision(agent, perception, json.loads(text))
        except Exception as exc:                      # noqa: BLE001
            self.errors += 1
            if self.on_error:
                self.on_error(agent, exc)
            # A failed call must not become a hidden nudge: fall back to the
            # agent's own local choice, exactly as if --llm were off.
            return agent.local_decide(perception, tick)

    def _to_decision(self, agent, perception, data):
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

    def __call__(self, agents, perceptions, tick):
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self._one, a, perceptions[a.id], tick)
                       for a in agents]
            return [(a, f.result()) for a, f in zip(agents, futures)]

    # -- cost ------------------------------------------------------------
    PRICES = {  # USD per million tokens, input / output
        "claude-opus-5": (5.0, 25.0),
        "claude-fable-5-1": (10.0, 50.0),
        "claude-sonnet-5": (2.0, 10.0),
        "claude-haiku-4-5": (1.0, 5.0),
    }

    def spend(self):
        pin, pout = self.PRICES.get(self.model, (5.0, 25.0))
        return (self.input_tokens * pin + self.output_tokens * pout) / 1e6

    @classmethod
    def estimate(cls, model, agents, ticks):
        """Rough, before you spend anything. ~900 in / ~120 out per call."""
        pin, pout = cls.PRICES.get(model, (5.0, 25.0))
        calls = agents * ticks
        return calls, calls * (900 * pin + 120 * pout) / 1e6
