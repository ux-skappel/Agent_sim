"""Where new utterances come from, in each of the two speech modes.

  tokens  invented syllables. Meaningless, and an unbounded pool -- any two
          agents saying the same thing means contact happened, never that
          they drew the same word from a shared bag. That is what makes
          convergence measurable.

  words   real words. Far more human, at a real cost to the experiment:
          the pool below is finite and it is *mine*. Every word in it is a
          concept the agents did not invent, and words are not neutral --
          "yes" and "no" alone hand them agreement and refusal for free.
          This is an acknowledged injection, not a clean slate. Read the
          list before drawing conclusions from a --speech words run.
"""

from .naming import make_word

# Chosen to be mundane and low-valence: nothing that names a goal, a threat,
# a reward, or a relationship worth wanting.
SEED_WORDS = [
    "stone", "water", "path", "light", "edge", "ground", "wind", "shadow",
    "circle", "line", "sound", "place", "thing", "side", "mark", "space",
    "move", "stop", "wait", "see", "hear", "come", "go", "stay",
    "turn", "hold", "give", "take", "make", "know", "think", "say",
    "here", "there", "now", "then", "again", "still", "away", "back",
    "near", "far", "small", "large", "quiet", "loud", "new", "old",
    "warm", "cold", "empty", "full", "same", "other", "many", "few",
    "i", "you", "we", "they", "this", "that", "not", "yes", "no", "maybe",
]

MODES = ("tokens", "words")


def coin(rng, mode):
    """A unit of speech this agent has not used before."""
    if mode == "words":
        return rng.choice(SEED_WORDS)
    return make_word(rng, 1, 2)
