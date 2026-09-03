"""Logging. Everything every agent does or says is written down."""

import json
import os
import time


class Recorder:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.events_path = os.path.join(run_dir, "events.jsonl")
        self.conv_path = os.path.join(run_dir, "conversations.log")
        self.frames_path = os.path.join(run_dir, "frames.json")
        self._events = open(self.events_path, "w", encoding="utf-8")
        self._conv = open(self.conv_path, "w", encoding="utf-8")
        self.frames = []
        self.started = time.time()

    def event(self, **fields):
        self._events.write(json.dumps(fields, ensure_ascii=False) + "\n")

    def speech(self, tick, speaker, tokens, listeners, target=None):
        line = "[t{:05d}] {} {} \"{}\"  ->  {}".format(
            tick, speaker,
            "to " + target if target else "says",
            " ".join(tokens),
            ", ".join(listeners) if listeners else "(nobody in range)")
        self._conv.write(line + "\n")

    def frame(self, tick, agents_state, speech_events):
        self.frames.append({"t": tick, "a": agents_state, "s": speech_events})

    def close(self, meta):
        self._events.close()
        self._conv.close()
        with open(self.frames_path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "frames": self.frames}, f,
                      separators=(",", ":"))
