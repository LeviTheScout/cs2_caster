"""Project paths and the pilot map list."""
import csv
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
META = DATA / "meta"
INTERIM = DATA / "interim"


@dataclass
class MapSpec:
    map_id: str
    event: str
    hltv_match_id: str
    stage: str
    map_name: str
    demo_files: list[Path]  # one file, or several parts of a split map in order
    vod_id: str
    caption_file: Path
    caster_crew: str


def pilot_maps() -> dict[str, MapSpec]:
    maps = {}
    with open(META / "pilot.tsv", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            maps[r["map_id"]] = MapSpec(
                map_id=r["map_id"],
                event=r["event"],
                hltv_match_id=r["hltv_match_id"],
                stage=r["stage"],
                map_name=r["map"],
                demo_files=[DATA / p for p in r["demo_files"].split(";")],
                vod_id=r["vod_id"],
                caption_file=DATA / r["caption_file"],
                caster_crew=r["caster_crew"],
            )
    return maps


def team_names() -> dict[str, str]:
    """Team name as written in demos -> one canonical name."""
    with open(META / "team_names.tsv", newline="") as f:
        return {r["demo_name"]: r["team"] for r in csv.DictReader(f, delimiter="\t")}
