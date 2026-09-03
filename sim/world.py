"""The closed environment.

A bounded grid with walls. Nothing in it but the agents: no food, no
resources, no hazards, no scoreboard. Time passes; that is all the world
provides.
"""

import random

from .agent import Agent, DIRS
from .naming import make_names


def _sign(v):
    return (v > 0) - (v < 0)


class World:
    def __init__(self, width=90, height=90, n_agents=100, vision=6,
                 hearing=6, seed=1, memory_capacity=500, decider=None,
                 speech_mode="tokens"):
        self.width = width
        self.height = height
        self.vision = vision
        self.hearing = hearing
        self.speech_mode = speech_mode
        self.tick_no = 0
        self.rng = random.Random(seed)
        # Optional replacement for how agents choose (see sim/llm_mind.py).
        # It receives one agent's own perception and nothing else.
        self.decider = decider

        names = make_names(self.rng, n_agents)
        self.agents = []
        for i, name in enumerate(names):
            a = Agent(i, name,
                      self.rng.randrange(width), self.rng.randrange(height),
                      random.Random(self.rng.randrange(2 ** 31)),
                      memory_capacity, speech_mode)
            self.agents.append(a)
        self.by_name = {a.name: a for a in self.agents}

        # A global token registry, used only for logging and colouring the
        # visualisation. No agent can read it.
        self.token_ids = {}
        # Bookkeeping for the human observer only.
        self.action_counts = {}
        self.utterance_count = 0
        self.addressed_counts = {}

    def token_id(self, token):
        if token not in self.token_ids:
            self.token_ids[token] = len(self.token_ids)
        return self.token_ids[token]

    # -- perception ------------------------------------------------------
    def _visible_to(self, agent):
        out = []
        for other in self.agents:
            if other is agent:
                continue
            dx = other.x - agent.x
            dy = other.y - agent.y
            if abs(dx) <= self.vision and abs(dy) <= self.vision:
                out.append({"name": other.name, "dx": dx, "dy": dy})
        return out

    def _in_range(self, agent, radius):
        out = []
        for other in self.agents:
            if other is agent:
                continue
            if (abs(other.x - agent.x) <= radius
                    and abs(other.y - agent.y) <= radius):
                out.append(other)
        return out

    # -- movement --------------------------------------------------------
    def _move(self, agent, dx, dy):
        agent.x = max(0, min(self.width - 1, agent.x + dx))
        agent.y = max(0, min(self.height - 1, agent.y + dy))

    def _step_toward(self, agent, tx, ty, away=False):
        dx = _sign(tx - agent.x)
        dy = _sign(ty - agent.y)
        if away:
            dx, dy = -dx, -dy
        if dx == 0 and dy == 0:
            dx, dy = agent.rng.choice(DIRS)
        self._move(agent, dx, dy)

    # -- one tick --------------------------------------------------------
    def step(self, recorder=None):
        self.tick_no += 1
        tick = self.tick_no

        # 1. Everyone perceives the world as it is at the start of the tick.
        perceptions = {}
        for a in self.agents:
            p = {"tick": tick, "here": (a.x, a.y), "visible": self._visible_to(a)}
            perceptions[a.id] = p
            a.memory.moments += 1
            a.perceive(p, tick)

        # 2. Everyone decides, privately and simultaneously.
        if self.decider is not None:
            decisions = self.decider(self.agents, perceptions, tick)
        else:
            decisions = [(a, a.decide(perceptions[a.id], tick))
                         for a in self.agents]

        # 3. Actions take effect.
        pending_speech = []
        for a, d in decisions:
            act = d["action"]
            if act == "wander":
                if a.rng.random() > 0.6:
                    a.heading = a.rng.choice(DIRS)
                self._move(a, *a.heading)
            elif act in ("approach", "avoid"):
                other = self.by_name.get(d["target"])
                if other is not None:
                    self._step_toward(a, other.x, other.y, away=(act == "avoid"))
            elif act == "observe":
                a.memory.record(tick, "observed",
                                who=[(v["name"], v["dx"], v["dy"])
                                     for v in perceptions[a.id]["visible"]][:8])
            elif act == "reflect":
                a.reflect(tick)
            elif act == "idle":
                a.memory.record(tick, "idle")
            elif act in ("speak", "address"):
                pending_speech.append((a, d))

            self.action_counts[act] = self.action_counts.get(act, 0) + 1

            if recorder is not None:
                ev = {"tick": tick, "agent": a.name, "action": act,
                      "x": a.x, "y": a.y}
                if "target" in d:
                    ev["target"] = d["target"]
                if "tokens" in d:
                    ev["tokens"] = d["tokens"]
                recorder.event(**ev)

        # 4. Speech is delivered to whoever happened to be in earshot.
        frame_speech = []
        for a, d in pending_speech:
            listeners = self._in_range(a, self.hearing)
            target = d.get("target")
            for lis in listeners:
                directed = (lis.name == target)
                lis.hear(tick, a.name, d["tokens"], directed=directed)
                if directed:
                    self.addressed_counts[lis.name] = \
                        self.addressed_counts.get(lis.name, 0) + 1
            a.memory.record(tick, "spoke", tokens=list(d["tokens"]),
                            target=target,
                            heard_by=[l.name for l in listeners][:8])
            self.utterance_count += 1
            for t in d["tokens"]:
                self.token_id(t)
            if recorder is not None:
                recorder.speech(tick, a.name, d["tokens"],
                                [l.name for l in listeners], target)
                frame_speech.append([a.id,
                                     self.by_name[target].id if target in self.by_name else -1,
                                     " ".join(d["tokens"])])

        # 5. Snapshot for the visualisation.
        if recorder is not None:
            state = []
            for a in self.agents:
                dom = a.memory.dominant_token()
                state.append([a.x, a.y,
                              self.token_id(dom) if dom is not None else -1])
            recorder.frame(tick, state, frame_speech)

    def run(self, ticks, recorder=None, on_tick=None):
        for _ in range(ticks):
            self.step(recorder)
            if on_tick is not None:
                on_tick(self)
