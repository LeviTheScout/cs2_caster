"""Read YouTube auto captions into a word table and fix names and game terms.

Auto captions mishear player names (m0NESY -> "Monosy", karrigan -> "Carrian")
and split or misspell game terms ("Tech 9", "op" for AWP). Fixes come only from
data/meta/aliases.tsv plus each name's own spellings, never from fuzzy guesses:
"Money" is one letter away from m0NESY, and a wrong fix would teach the model
to mention a player who was never there. `suggest` lists likely misses so a
person can add them.

Rules:
- Player and team aliases apply only to the teams on that VOD. Other players
  and teams match only by their own name.
- An alias that is an ordinary English word ("hunter", "magic", "star") only
  counts when the caption capitalises it mid-sentence. At a sentence start it
  is tagged with sure=False and the text is left alone.
- An alias written with capitals in aliases.tsv ("HE") must match that case.
- Kind "english" in aliases.tsv marks slang the system word list lacks ("bro").
- A fix only corrects spelling. It never adds a word the caster did not say.
"""
import csv
import json
import re
from pathlib import Path

import pandas as pd

from .paths import META

DICT_FILE = Path("/usr/share/dict/words")
# Leetspeak in player names, read as letters: m0NESY -> monesy, sh1ro -> shiro.
LEET = str.maketrans("013457", "oieast")
NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
SENTENCE_END = (".", "?", "!")
# Kinds whose aliases can be ordinary words and need the capital-letter check.
NAME_KINDS = {"player", "team", "map"}
# Kinds that take a plural "s" ("AKs", "smokes", "CTs").
PLURAL_KINDS = {"weapon", "utility", "side"}
MAX_ALIAS_WORDS = 3


def english_words() -> set[str] | None:
    """Lowercase dictionary words, or None when the system has no word list."""
    if not DICT_FILE.exists():
        return None
    return {w.strip() for w in DICT_FILE.open(encoding="utf-8", errors="ignore") if w.strip().islower()}


def key(s: str) -> str:
    """Lowercase letters and digits only. A dash between digits stays, so the score 9-10 is not the player 910."""
    s = re.sub(r"(?<=\d)[-/](?=\d)", "~", s.lower())
    return re.sub(r"[^a-z0-9~]", "", s)


def name_forms(name: str) -> list[tuple[str, ...]]:
    """The ways a name can appear when spoken: as written, leetspeak read as letters, CamelCase split.

    A leading "The" is dropped, so "the Mongolz" only fixes the team's word.
    """
    words = re.sub(r"^The ", "", name).split()
    plain = tuple(key(w) for w in words)
    forms = {plain, tuple(k.translate(LEET) for k in plain)}
    parts = re.findall(r"[A-Z]+[a-z]*|[a-z]+", name)
    if len(words) == 1 and len(parts) > 1 and all(len(p) >= 3 for p in parts) and "".join(parts) == name:
        forms.add(tuple(p.lower() for p in parts))
    return [f for f in forms if all(f)]


def read_words(path: Path) -> pd.DataFrame:
    """One row per caption token: time in seconds, the text, and whether a new speaker starts (">>")."""
    rows, speaker = [], False
    tag = None  # a sound tag split over tokens, like "[ __ ]"
    for e in json.loads(path.read_text())["events"]:
        for s in e.get("segs") or []:
            t = (e.get("tStartMs", 0) + s.get("tOffsetMs", 0)) / 1000
            for w in s.get("utf8", "").split():
                if w == ">>":
                    speaker = True
                    continue
                if tag is not None:
                    tag[1] += " " + w
                    if "]" in w:
                        rows.append(tag + [speaker, True]); tag, speaker = None, False
                    continue
                if w.startswith("[") and "]" not in w:
                    tag = [t, w]
                    continue
                rows.append([t, w, speaker, w.startswith("[")])
                speaker = False
    return pd.DataFrame(rows, columns=["t", "raw", "speaker", "sound"])


def split_token(raw: str) -> tuple[str, str, str]:
    """'"Nico's,' -> ('"', 'Nico', "'s,"): leading punctuation, the word, and its possessive and punctuation."""
    m = re.match(r"^(\W*)(.*?)((?:['’]s?)?\W*)$", raw)
    return m.group(1), m.group(2), m.group(3)


class Vocab:
    """Every alias a caption span can match, for the teams on one VOD."""

    def __init__(self, rows: list[dict], english: set[str] | None):
        self.english = english
        self.table: dict[tuple[str, ...], list[dict]] = {}
        for r in rows:
            self.table.setdefault(r["keys"], []).append(r)

    @classmethod
    def build(cls, aliases: list[dict], players: dict[str, str], team_names: dict[str, str],
              roster: dict[str, list[str]], english: set[str] | None) -> "Vocab":
        """`roster` is team -> steam ids of the players on this VOD."""
        on_vod = {sid for ids in roster.values() for sid in ids}
        by_name = {n: sid for sid, n in players.items()}
        # Slang the word list lacks ("bro") is declared in aliases.tsv with kind "english".
        if english is not None:
            english = english | {key(a["alias"]) for a in aliases if a["kind"] == "english"}
        aliases = [a for a in aliases if a["kind"] != "english"]
        rows = []

        def add(kind, entity, keys, write, case=None, steamid=None):
            rows.append({"kind": kind, "entity": entity, "keys": keys, "write": write, "case": case, "steamid": steamid})

        for sid, name in players.items():
            for f in name_forms(name):
                add("player", name, f, name, steamid=sid)
        teams = set(team_names.values())
        for team in teams:
            for f in name_forms(team):
                add("team", team, f, re.sub(r"^The ", "", team))
        for a in aliases:
            kind, entity, alias = a["kind"], a["entity"], a["alias"]
            if kind == "player":
                if entity not in by_name:
                    raise ValueError(f"aliases.tsv: player {entity!r} is not in players.tsv")
                if by_name[entity] not in on_vod:
                    continue
            if kind == "team":
                if entity not in teams:
                    raise ValueError(f"aliases.tsv: team {entity!r} is not in team_names.tsv")
                if entity not in roster:
                    continue
            keys = tuple(key(w) for w in alias.split())
            case = tuple(alias.split()) if alias != alias.lower() else None
            add(kind, entity, keys, a["write"] or (entity if kind in NAME_KINDS else alias),
                case, by_name.get(entity) if kind == "player" else None)
        # Players and teams not on this VOD keep only their own names, and only if not an English word.
        rows = [r for r in rows if not (
            (r["kind"] == "player" and r["steamid"] not in on_vod or r["kind"] == "team" and r["entity"] not in roster)
            and (english is None or len(r["keys"]) == 1 and r["keys"][0] in english))]
        return cls(rows, english)

    def is_english(self, keys: tuple[str, ...]) -> bool:
        return self.english is None or len(keys) == 1 and keys[0] in self.english

    def match(self, cores: list[str], at: int, limit: int) -> tuple[dict, int] | None:
        """The longest alias starting at token `at` and not running past `limit`."""
        for n in range(min(MAX_ALIAS_WORDS, limit - at), 0, -1):
            words = cores[at:at + n]
            keys = tuple(key(w) for w in words)
            for r in self.table.get(keys, []):
                if r["case"] is None or tuple(words) == r["case"]:
                    return r, n
        return None

    def match_plural(self, core: str) -> tuple[dict, str] | None:
        """'AKs' -> the AK alias plus the ending 's'."""
        for end in ("es", "s"):
            if core.lower().endswith(end) and len(core) > len(end) + 1:
                stem = core[:-len(end)]
                for r in self.table.get((key(stem),), []):
                    if r["kind"] in PLURAL_KINDS and (r["case"] is None or (stem,) == r["case"]):
                        return r, core[-len(end):]
        return None


def _count(cores: list[str], at: int, limit: int) -> tuple[str, int] | None:
    """Player counts written as 'XvY': '1 V 3', 'five versus two', '5-v-4', '1 v one'."""
    def num(w):
        w = key(w)
        return int(w) if w in {"1", "2", "3", "4", "5"} else NUMBERS.get(w)

    ks = [key(w) for w in cores[at:min(at + 3, limit)]]
    if ks and (m := re.fullmatch(r"([1-5])v([1-5])", ks[0])):
        return f"{m[1]}v{m[2]}", 1
    if len(ks) >= 2 and num(cores[at]):
        if len(ks) >= 3 and ks[1] in ("v", "vs", "versus") and num(cores[at + 2]):
            return f"{num(cores[at])}v{num(cores[at + 2])}", 3
        if m := re.fullmatch(r"v([1-5])", ks[1]):
            return f"{num(cores[at])}v{m[1]}", 2
    if len(ks) >= 2 and (m := re.fullmatch(r"([1-5])v", ks[0])) and num(cores[at + 1]):
        return f"{m[1]}v{num(cores[at + 1])}", 2
    return None


def _cased(write: str, first_core: str) -> str:
    """A term keeps the caption's shouting ('SMOKE') and a lowercase term its sentence-start capital ('Smoke')."""
    if len(first_core) > 1 and first_core.isupper():
        return write.upper()
    if write.islower() and first_core[:1].isupper():
        return write[0].upper() + write[1:]
    return write


def clean(words: pd.DataFrame, vocab: Vocab) -> pd.DataFrame:
    """Merge alias spans into one row each and fix their spelling.

    Columns: t, raw (what the caption said), text (fixed), speaker, sound,
    kind (player/team/map/weapon/utility/side/count or empty), entity,
    steamid (players), sure (False: an English-word alias at a sentence start, text left as is).
    """
    raws = words.raw.tolist()
    pre, cores, post = zip(*(split_token(w) for w in raws)) if raws else ((), (), ())
    cores = list(cores)
    sound, speaker = words.sound.tolist(), words.speaker.tolist()
    n = len(raws)
    # A span can't cross a new speaker, a sound tag, or punctuation inside it
    # ("9, 10" is not the player 910, "CT's side" is not "CT side"). A dash can: "T- side".
    limit, end = [0] * n, n
    for i in range(n - 1, -1, -1):
        if sound[i]:
            end = i
        limit[i] = end
        if i > 0 and (speaker[i] or sound[i] or post[i - 1] not in ("", "-")):
            end = i
    out = []
    i = 0
    while i < n:
        row = {"t": words.t.iat[i], "raw": raws[i], "text": raws[i], "speaker": speaker[i], "sound": sound[i],
               "kind": "", "entity": "", "steamid": "", "sure": True}
        if sound[i] or not cores[i]:
            out.append(row); i += 1
            continue
        start_of_sentence = i == 0 or speaker[i] or sound[i - 1] or raws[i - 1].endswith(SENTENCE_END)
        found = vocab.match(cores, i, limit[i])
        width, ending = 1, ""
        if found:
            r, width = found
        elif (c := _count(cores, i, limit[i])):
            r, width = {"kind": "count", "entity": c[0], "write": c[0], "steamid": None, "keys": ()}, c[1]
        elif (p := vocab.match_plural(cores[i])):
            r, ending = p
            if r["write"].lower().endswith("s"):  # "Famass" is FAMAS, not FAMASs
                ending = ""
        else:
            out.append(row); i += 1
            continue
        if r["kind"] in NAME_KINDS and vocab.is_english(r["keys"]):
            if not cores[i][:1].isupper():
                out.append(row); i += 1
                continue
            row["sure"] = not start_of_sentence
        span = raws[i:i + width]
        row.update(raw=" ".join(span), kind=r["kind"], entity=r["entity"], steamid=r["steamid"] or "")
        if row["sure"]:
            write = r["write"] if r["kind"] in NAME_KINDS else _cased(r["write"], cores[i])
            row["text"] = pre[i] + write + ending + post[i + width - 1]
        else:
            row["text"] = row["raw"]
        out.append(row)
        i += width
    return pd.DataFrame(out)


def read_aliases(path: Path = META / "aliases.tsv") -> list[dict]:
    """Rows of kind, entity, alias, write. Lines starting with # are comments."""
    with open(path, newline="") as f:
        lines = [l for l in f if l.strip() and not l.startswith("#")]
    rows = list(csv.DictReader(lines, delimiter="\t"))
    for r in rows:
        r["write"] = (r.get("write") or "").strip()
    return rows
