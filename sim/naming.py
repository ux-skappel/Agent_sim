"""Deterministic, unique identity generation.

Names carry no meaning and imply no role. They exist only so agents can
refer to each other and so a human can read the logs.
"""

CONSONANTS = "bdfgkhlmnprstvz"
VOWELS = "aeiou"


def syllable(rng):
    return rng.choice(CONSONANTS) + rng.choice(VOWELS)


def make_word(rng, min_syl=1, max_syl=3):
    return "".join(syllable(rng) for _ in range(rng.randint(min_syl, max_syl)))


def make_names(rng, count):
    """Return `count` unique, capitalised names."""
    names = []
    seen = set()
    while len(names) < count:
        w = make_word(rng, 2, 3)
        if w in seen:
            continue
        seen.add(w)
        names.append(w.capitalize())
    return names
