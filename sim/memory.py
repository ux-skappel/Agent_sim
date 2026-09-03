"""Per-agent private memory.

Everything in here got written by the agent's own senses. Nothing is shared
between agents, and nothing here is a score the agent tries to raise -- these
are records, not rewards.
"""

from collections import deque


class Memory:
    def __init__(self, capacity=300):
        self.episodes = deque(maxlen=capacity)   # what happened, in order
        self.lexicon = {}                        # token -> times encountered
        self.acquaintances = {}                  # name -> private impression
        self.self_note = None                    # what it last said it was
        self.moments = 0                         # how long it has existed

    # -- writing ---------------------------------------------------------
    def record(self, tick, kind, **fields):
        entry = {"tick": tick, "kind": kind}
        entry.update(fields)
        self.episodes.append(entry)
        return entry

    def note_token(self, token, weight=1):
        self.lexicon[token] = self.lexicon.get(token, 0) + weight

    def note_agent(self, name, tick, seen=0, heard=0, addressed=0):
        a = self.acquaintances.get(name)
        if a is None:
            a = {"seen": 0, "heard": 0, "addressed_me": 0, "last_tick": tick,
                 "first_tick": tick}
            self.acquaintances[name] = a
        a["seen"] += seen
        a["heard"] += heard
        a["addressed_me"] += addressed
        a["last_tick"] = tick

    # -- reading ---------------------------------------------------------
    def salience(self, name, tick):
        """How present someone is in this agent's mind. Not a preference:
        approach and avoid both draw from the same number."""
        a = self.acquaintances.get(name)
        if a is None:
            return 0.0
        contact = a["seen"] + 2 * a["heard"] + 3 * a["addressed_me"]
        recency = 1.0 / (1.0 + 0.02 * max(0, tick - a["last_tick"]))
        return contact * recency

    def known_names(self):
        return list(self.acquaintances.keys())

    def dominant_token(self):
        if not self.lexicon:
            return None
        return max(self.lexicon.items(), key=lambda kv: (kv[1], kv[0]))[0]

    def recent(self, n=12):
        return list(self.episodes)[-n:]

    def recent_speech(self, n=10):
        """Conversation only. Outlives the general window, because what was
        said to you is the part of the past that keeps mattering."""
        talk = [e for e in self.episodes if e["kind"] in ("heard", "spoke")]
        return talk[-n:]

    def note_self(self, text):
        text = (text or "").strip()
        if text:
            self.self_note = text[:280]

    def snapshot(self):
        return {
            "self_note": self.self_note,
            "moments": self.moments,
            "lexicon": dict(sorted(self.lexicon.items(),
                                   key=lambda kv: -kv[1])[:20]),
            "acquaintances": {
                k: dict(v) for k, v in sorted(
                    self.acquaintances.items(),
                    key=lambda kv: -(kv[1]["seen"] + kv[1]["heard"]))[:15]
            },
            "recent": self.recent(15),
        }
