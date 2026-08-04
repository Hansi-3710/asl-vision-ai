"""
vocab.py
========
Word/gloss <-> integer id mapping for the CTC decoder. Index 0 is always
the CTC blank token (never a real word) -- this matches
torch.nn.CTCLoss's default `blank=0` and must stay index 0 everywhere a
vocab is loaded, training or inference.

A real deployment builds this from the WLASL/How2Sign class list (one
entry per gloss). Until that dataset is wired in, `SYNTHETIC_GLOSSES`
below is a small placeholder vocabulary used only by
`generate_synthetic_dataset.py` and the bundled smoke-test checkpoint --
see that module's docstring for why a synthetic dataset exists at all.
"""

from __future__ import annotations

import json
from pathlib import Path

BLANK_TOKEN = "<blank>"
UNK_TOKEN = "<unk>"

# Placeholder-only vocabulary for the synthetic smoke-test pipeline. NOT
# real ASL glosses in any linguistically meaningful sense -- just 12
# distinct labels used to prove the train -> export -> serve path works.
SYNTHETIC_GLOSSES = [
    "HELLO", "MY", "NAME", "IS", "YOU", "LIKE",
    "THANK", "PLEASE", "YES", "NO", "HELP", "GOOD",
]


class Vocabulary:
    def __init__(self, words: list[str]):
        if BLANK_TOKEN in words or UNK_TOKEN in words:
            raise ValueError("`words` should contain only real glosses -- blank/unk are added automatically.")
        # Sorted for a deterministic id assignment regardless of input order
        # (important for reproducibility: two calls to build_vocab on the
        # same word set must always produce the same ids).
        deduped_sorted = sorted(set(words))
        self.itos: list[str] = [BLANK_TOKEN, UNK_TOKEN] + deduped_sorted
        self.stoi: dict[str, int] = {w: i for i, w in enumerate(self.itos)}

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, gloss: str) -> int:
        return self.stoi.get(gloss.upper(), self.unk_id)

    def encode_sequence(self, glosses: list[str]) -> list[int]:
        return [self.encode(g) for g in glosses]

    def decode_id(self, idx: int) -> str:
        if idx < 0 or idx >= len(self.itos):
            return UNK_TOKEN
        return self.itos[idx]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"words": self.itos[2:]}, f, indent=2)  # persist only real words, ids are re-derived

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(data["words"])


def build_synthetic_vocab() -> Vocabulary:
    return Vocabulary(SYNTHETIC_GLOSSES)


if __name__ == "__main__":
    vocab = build_synthetic_vocab()
    assert vocab.decode_id(vocab.blank_id) == BLANK_TOKEN
    assert vocab.itos[0] == BLANK_TOKEN and vocab.itos[1] == UNK_TOKEN
    assert len(vocab) == len(SYNTHETIC_GLOSSES) + 2

    roundtrip_ids = vocab.encode_sequence(["hello", "my", "name"])
    roundtrip_words = [vocab.decode_id(i) for i in roundtrip_ids]
    assert roundtrip_words == ["HELLO", "MY", "NAME"], roundtrip_words

    assert vocab.encode("not-a-real-gloss") == vocab.unk_id

    tmp_path = "/tmp/_vocab_roundtrip_check.json"
    vocab.save(tmp_path)
    reloaded = Vocabulary.load(tmp_path)
    assert reloaded.itos == vocab.itos, "Vocabulary did not round-trip through JSON!"

    print(f"vocab.py OK -- {len(vocab)} tokens (incl. blank/unk), round-trips through JSON correctly.")
