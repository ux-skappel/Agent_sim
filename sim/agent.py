"""An agent: an identity, a private memory, and a way of choosing.

Design rules this file obeys:
  * No goal, role, reward, utility or fitness of any kind.
  * Every affiliative action has an equally-weighted opposite
    (approach <-> avoid, speak <-> idle), so nothing pushes the population
    toward sociality or away from it.
  * Temperament is random at birth. It makes agents *different*, it does not
    make them *want* anything.
  * An agent may only use what is in its own memory and its own perception.
"""

from .memory import Memory
from .naming import make_word

ACTIONS = ["idle", "wander", "approach", "avoid", "observe",
           "speak", "address", "reflect"]

DIRS = [(-1, -1), (0, -1), (1, -1), (-1, 0),
        (1, 0), (-1, 1), (0, 1), (1, 1)]


class Agent:
    def __init__(self, agent_id, name, x, y, rng, memory_capacity=300):
        self.id = agent_id
        self.name = name
        self.x = x
        self.y = y
        self.rng = rng
        self.memory = Memory(memory_capacity)
        self.heading = rng.choice(DIRS)
        self.alive_since = 0

        # Identity, not motivation: an independent draw per action type,
        # from the same distribution for every action, then normalised.
        raw = [rng.expovariate(1.0) for _ in ACTIONS]
        total = sum(raw)
        self.temperament = {a: r / total for a, r in zip(ACTIONS, raw)}

        # How readily this agent coins a new sound instead of reusing one.
        self.invent_rate = rng.uniform(0.02, 0.25)
        # How many sounds it tends to put in one utterance.
        self.verbosity = rng.randint(1, 3)

    # -- speech ----------------------------------------------------------
    def _pick_token(self):
        lex = self.memory.lexicon
        if not lex or self.rng.random() < self.invent_rate:
            token = make_word(self.rng, 1, 2)
            self.memory.note_token(token)
            return token
        tokens = list(lex.keys())
        weights = [lex[t] for t in tokens]
        return self.rng.choices(tokens, weights=weights, k=1)[0]

    def utterance(self, tick):
        n = max(1, min(3, self.rng.randint(1, self.verbosity)))
        tokens = [self._pick_token() for _ in range(n)]
        # Sometimes an agent says a name it remembers. Nothing rewards this;
        # it is simply another thing in memory that can come out of the mouth.
        known = self.memory.known_names()
        if known and self.rng.random() < 0.18:
            weights = [self.memory.salience(k, tick) + 0.01 for k in known]
            tokens.append(self.rng.choices(known, weights=weights, k=1)[0])
        return tokens

    # -- choosing --------------------------------------------------------
    def _pick_target(self, visible, tick):
        """Choose someone to act on. Salience is symmetric: the same number
        feeds approach and avoid."""
        if not visible:
            return None
        names = [v["name"] for v in visible]
        weights = [self.memory.salience(n, tick) + 0.5 for n in names]
        return self.rng.choices(names, weights=weights, k=1)[0]

    def decide(self, perception, tick):
        visible = perception["visible"]
        weights = dict(self.temperament)
        if not visible:
            for a in ("approach", "avoid", "address"):
                weights[a] = 0.0
        options = [a for a in ACTIONS if weights[a] > 0]
        if not options:
            return {"action": "idle"}
        choice = self.rng.choices(options,
                                  weights=[weights[a] for a in options], k=1)[0]

        if choice in ("approach", "avoid"):
            target = self._pick_target(visible, tick)
            if target is None:
                return {"action": "wander"}
            return {"action": choice, "target": target}
        if choice == "address":
            target = self._pick_target(visible, tick)
            if target is None:
                return {"action": "speak", "tokens": self.utterance(tick)}
            return {"action": "address", "target": target,
                    "tokens": self.utterance(tick)}
        if choice == "speak":
            return {"action": "speak", "tokens": self.utterance(tick)}
        return {"action": choice}

    # -- perceiving ------------------------------------------------------
    def perceive(self, perception, tick):
        for v in perception["visible"]:
            self.memory.note_agent(v["name"], tick, seen=1)
        if perception["visible"]:
            self.memory.record(
                tick, "saw",
                who=[v["name"] for v in perception["visible"]][:8])

    def hear(self, tick, speaker, tokens, directed):
        for t in tokens:
            self.memory.note_token(t)
        self.memory.note_agent(speaker, tick, heard=1,
                               addressed=1 if directed else 0)
        self.memory.record(tick, "heard", speaker=speaker,
                           tokens=list(tokens), directed=directed)

    def reflect(self, tick):
        """Re-read your own memory. Sometimes a sound you have been carrying
        gets a little heavier just from being thought about."""
        lex = self.memory.lexicon
        if lex:
            tokens = list(lex.keys())
            weights = [lex[t] for t in tokens]
            t = self.rng.choices(tokens, weights=weights, k=1)[0]
            self.memory.note_token(t)
            self.memory.record(tick, "reflected", token=t)
        else:
            self.memory.record(tick, "reflected", token=None)
