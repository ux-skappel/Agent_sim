#!/usr/bin/env python3
"""Checks on the properties this simulation claims to have."""

import unittest

from sim.world import World
from sim.agent import ACTIONS


def run(seed=3, ticks=40, **kw):
    w = World(n_agents=30, width=40, height=40, seed=seed, **kw)
    w.run(ticks)
    return w


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_world(self):
        a, b = run(seed=5), run(seed=5)
        self.assertEqual([(x.name, x.x, x.y) for x in a.agents],
                         [(x.name, x.x, x.y) for x in b.agents])

    def test_different_seed_different_world(self):
        a, b = run(seed=5), run(seed=6)
        self.assertNotEqual([(x.x, x.y) for x in a.agents],
                            [(x.x, x.y) for x in b.agents])


class TestIdentity(unittest.TestCase):
    def test_names_unique(self):
        w = run()
        self.assertEqual(len(set(a.name for a in w.agents)), len(w.agents))

    def test_memories_are_separate_objects(self):
        w = run()
        ids = {id(a.memory) for a in w.agents}
        self.assertEqual(len(ids), len(w.agents))

    def test_temperaments_differ(self):
        w = run()
        first = w.agents[0].temperament
        self.assertTrue(any(a.temperament != first for a in w.agents[1:]))


class TestPrivateKnowledge(unittest.TestCase):
    def test_agent_never_knows_itself_as_an_acquaintance(self):
        w = run(ticks=60)
        for a in w.agents:
            self.assertNotIn(a.name, a.memory.acquaintances,
                             "%s recorded itself as an acquaintance" % a.name)

    def test_nobody_knows_everybody(self):
        """With a sparse world and a short run, knowledge must be partial --
        i.e. it is coming from perception, not from a global table."""
        w = World(n_agents=30, width=80, height=80, seed=11)
        w.run(30)
        for a in w.agents:
            self.assertLess(len(a.memory.acquaintances), len(w.agents) - 1)

    def test_lexicon_only_holds_heard_or_invented_sounds(self):
        """Every sound in a memory must be a sound that actually got uttered
        somewhere in the world (or coined by that agent itself)."""
        w = run(ticks=60)
        uttered = set(w.token_ids)
        names = {a.name for a in w.agents}
        for a in w.agents:
            unexplained = set(a.memory.lexicon) - uttered - names
            # An agent may coin a sound while reflecting and never say it.
            for tok in unexplained:
                self.assertEqual(a.memory.lexicon[tok], 1, tok)

    def test_memory_is_capped(self):
        w = World(n_agents=20, width=30, height=30, seed=2, memory_capacity=25)
        w.run(80)
        for a in w.agents:
            self.assertLessEqual(len(a.memory.episodes), 25)


class TestNoHiddenObjective(unittest.TestCase):
    def test_no_reward_machinery_in_the_source(self):
        """No agent may carry a score, and no code may compute one."""
        import os
        banned = ("reward", "utility", "fitness", "payoff", "score",
                  "objective", "incentive")
        for root, _, files in os.walk("sim"):
            for fn in sorted(files):
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(root, fn)
                with open(path, encoding="utf-8") as f:
                    code = "\n".join(line.split("#")[0]
                                     for line in f).lower()
                for word in banned:
                    self.assertNotIn("self." + word, code,
                                     "%s carries a %s" % (path, word))
                    self.assertNotIn("def " + word, code,
                                     "%s computes a %s" % (path, word))

    def test_approach_and_avoid_are_equally_likely_overall(self):
        """Across the population, nothing tilts toward or away from company."""
        w = World(n_agents=100, width=60, height=60, seed=4)
        w.run(200)
        ap = w.action_counts.get("approach", 0)
        av = w.action_counts.get("avoid", 0)
        ratio = ap / max(1, av)
        self.assertGreater(ratio, 0.75)
        self.assertLess(ratio, 1.35)

    def test_every_action_including_doing_nothing_gets_used(self):
        w = World(n_agents=100, width=60, height=60, seed=4)
        w.run(100)
        for act in ACTIONS:
            self.assertGreater(w.action_counts.get(act, 0), 0, act)


class TestLLMMindPlumbing(unittest.TestCase):
    """The --llm path, exercised without spending anything."""

    def _agent_and_view(self):
        w = World(n_agents=6, width=12, height=12, seed=1)
        w.run(15)
        a = w.agents[0]
        return a, {"tick": 16, "here": (a.x, a.y),
                   "visible": [{"name": w.agents[1].name, "dx": 2, "dy": -1}]}

    def test_prompt_leaks_nothing_the_agent_has_not_perceived(self):
        from sim.llm_mind import _describe
        w = World(n_agents=8, width=60, height=60, seed=2)
        w.run(10)
        a = w.agents[0]
        view = {"tick": 11, "here": (a.x, a.y), "visible": []}
        text = _describe(a, view, 11)
        allowed = set(a.memory.acquaintances) | {a.name}
        for other in w.agents:
            if other.name not in allowed:
                self.assertNotIn(other.name, text,
                                 "prompt named a stranger: %s" % other.name)

    def test_unseen_target_cannot_be_acted_on(self):
        from sim.llm_mind import LLMMinds
        a, view = self._agent_and_view()
        d = LLMMinds._to_decision(None, a, view,
                                  {"action": "approach", "target": "Ghost",
                                   "utterance": None})
        self.assertEqual(d["action"], "wander")

    def test_actions_map_through(self):
        from sim.llm_mind import LLMMinds
        a, view = self._agent_and_view()
        seen = view["visible"][0]["name"]
        cases = [
            ({"action": "idle", "target": None, "utterance": None},
             {"action": "idle"}),
            ({"action": "avoid", "target": seen, "utterance": None},
             {"action": "avoid", "target": seen}),
            ({"action": "address", "target": seen, "utterance": "ka ru"},
             {"action": "address", "tokens": ["ka", "ru"], "target": seen}),
            ({"action": "fly", "target": None, "utterance": None},
             {"action": "idle"}),
        ]
        for data, want in cases:
            self.assertEqual(LLMMinds._to_decision(None, a, view, data), want)

    def test_utterance_is_capped(self):
        from sim.llm_mind import LLMMinds
        a, view = self._agent_and_view()
        d = LLMMinds._to_decision(None, a, view,
                                  {"action": "speak", "target": None,
                                   "utterance": "a b c d e f g"})
        self.assertEqual(len(d["tokens"]), 4)

    def test_cost_estimate_is_reported_before_spending(self):
        from sim.llm_mind import LLMMinds
        calls, cost = LLMMinds.estimate("claude-opus-5", 100, 600)
        self.assertEqual(calls, 60000)
        self.assertGreater(cost, 0)


class TestWorldIsClosed(unittest.TestCase):
    def test_nobody_leaves(self):
        w = World(n_agents=40, width=20, height=20, seed=8)
        w.run(120)
        for a in w.agents:
            self.assertTrue(0 <= a.x < 20 and 0 <= a.y < 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
