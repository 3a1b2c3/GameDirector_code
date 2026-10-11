#!/usr/bin/env python3
"""Fast CPU-side integrity and custom-combat checks for the release bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "agentic_control"))

from closed_loop.combat import CombatState, DamageCalculator  # noqa: E402
from closed_loop.games import get_game  # noqa: E402
from closed_loop.prompts import PlayerPromptTrack, PromptVocabulary  # noqa: E402


EXPECTED_SMALL = {
    "attacknet": Path(os.environ.get(
        "GAMEDIRECTOR_ATTACK_CHECKPOINT",
        REPOSITORY_ROOT / "model" / "attacknet" / "attacknet.pt",
    )).expanduser(),
    "situationnet": Path(os.environ.get(
        "GAMEDIRECTOR_SITUATION_CHECKPOINT",
        REPOSITORY_ROOT / "model" / "situationnet" / "situationnet.pt",
    )).expanduser(),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hash", action="store_true", help="also hash the 9.4 GB world model")
    parser.add_argument(
        "--world-model-checkpoint",
        type=Path,
        default=Path(os.environ.get(
            "GAMEDIRECTOR_WORLD_MODEL_CHECKPOINT",
            REPOSITORY_ROOT / "model" / "world_model" / "ar_world_model.safetensors",
        )),
        help="causal AR checkpoint (or set GAMEDIRECTOR_WORLD_MODEL_CHECKPOINT)",
    )
    args = parser.parse_args()
    expected = {**EXPECTED_SMALL, "world_model_ar": args.world_model_checkpoint.expanduser()}

    missing = [str(path) for path in expected.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing bundled model files: {missing}")

    for name in ("attacknet", "situationnet"):
        config_path = expected[name].with_name("config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise SystemExit(f"invalid config: {config_path}")

    from safetensors import safe_open

    tensor_counts = {}
    for name in ("world_model_ar",):
        with safe_open(expected[name], framework="pt", device="cpu") as handle:
            tensor_counts[name] = len(list(handle.keys()))
        if tensor_counts[name] != 825:
            raise SystemExit(
                f"unexpected {name} tensor count: {tensor_counts[name]} (expected 825)"
            )

    state = CombatState(
        player_hp=777,
        boss_hp=1234,
        player_attack=73,
        boss_attack=29,
        boss_skill=37,
        boss_skill_max=37,
    )
    state.spend_skill()
    assert state.boss_skill == 0
    cards = get_game("nrftw").skill_cards("Golden Bird", skill_cost=37)
    assert cards and {card["cost"] for card in cards.values()} == {37}
    result = DamageCalculator.apply_cell(
        state,
        player_hit=True,
        boss_hit=True,
        player_attack_prompt=True,
        boss_attack_prompt=True,
        skill_gain_per_hit=3,
    )
    assert state.player_hp == 748 and state.boss_hp == 1161
    assert state.boss_skill == 6 and result["skill_gain"] == 6

    examples = REPOSITORY_ROOT / "example" / "manifest.json"
    example_count = 0
    if examples.is_file():
        manifest = json.loads(examples.read_text(encoding="utf-8"))
        for row in manifest["samples"]:
            sample_dir = REPOSITORY_ROOT / "example" / row["game"] / row["name"]
            game = get_game(row["game"])
            vocabulary = PromptVocabulary.from_snapshot(
                sample_dir / "vocabulary.json",
                game=game,
                player_name=row["player"],
                boss_name=row["boss"],
                background=row["background"],
            )
            track = PlayerPromptTrack.from_path(sample_dir / "player_track.json", vocabulary)
            clauses, wrapped = track.chunk(0)
            if len(clauses) != 20 or wrapped:
                raise SystemExit(f"invalid first prompt chunk: {row['name']}")
            if not (sample_dir / "first_frame.png").is_file():
                raise SystemExit(f"missing first frame: {row['name']}")
            example_count += 1

    hashes = {}
    if args.hash:
        hashes = {name: sha256(path) for name, path in expected.items()}
    print(json.dumps({
        "ok": True,
        "world_model_tensors": tensor_counts,
        "example_count": example_count,
        "hashes": hashes,
    }, indent=2))


if __name__ == "__main__":
    main()
