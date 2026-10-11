#!/usr/bin/env python3
"""Run one real closed-loop agent for NRFTW, Vampire, or Hollow Knight."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from closed_loop.combat import CombatState  # noqa: E402
from closed_loop.components import (  # noqa: E402
    ARWorldModelRunner,
    AttackNetRunner,
    SituationNetRunner,
    WorldModelRunner,
)
from closed_loop.decision import make_runtime_decision_components  # noqa: E402
from closed_loop.games import ALL_SKILL_NAMES, GAMES, get_game  # noqa: E402
from closed_loop.loop import ClosedLoopAgent, ClosedLoopAgentAR  # noqa: E402
from closed_loop.prompts import PlayerPromptTrack, PromptVocabulary, detect_background  # noqa: E402
from llm.service import DEFAULT_CHECKPOINT as DEFAULT_LLM_CHECKPOINT  # noqa: E402
from llm.service import ensure_llm_server  # noqa: E402


OUTPUT_ROOT = REPOSITORY_ROOT / "output"
DEFAULT_ATTACK = Path(os.environ.get(
    "GAMEDIRECTOR_ATTACK_CHECKPOINT",
    REPOSITORY_ROOT / "model" / "attacknet" / "attacknet.pt",
)).expanduser()
DEFAULT_SITUATION = Path(os.environ.get(
    "GAMEDIRECTOR_SITUATION_CHECKPOINT",
    REPOSITORY_ROOT / "model" / "situationnet" / "situationnet.pt",
)).expanduser()
DEFAULT_WORLD_MODEL = Path(os.environ.get(
    "GAMEDIRECTOR_WORLD_MODEL_CHECKPOINT",
    REPOSITORY_ROOT / "model" / "world_model" / "bidirectional_world_model.safetensors",
)).expanduser()
DEFAULT_AR_WORLD_MODEL = Path(os.environ.get(
    "GAMEDIRECTOR_WORLD_MODEL_CHECKPOINT",
    REPOSITORY_ROOT / "model" / "world_model" / "ar_world_model.safetensors",
)).expanduser()
DEFAULT_BASE_MODEL = Path(os.environ.get(
    "GAMEDIRECTOR_BASE_MODEL", REPOSITORY_ROOT / "model",
)).expanduser()


def default_output_dir(args: argparse.Namespace) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [args.game, "rollout", stamp]
    if args.ticks != 12:
        parts.append(f"{args.ticks * 5}s")
    if args.player_choice != 1:
        parts.append(f"p{args.player_choice}")
    if args.boss_choice != 1:
        parts.append(f"b{args.boss_choice}")
    return OUTPUT_ROOT / "_".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--game", choices=tuple(GAMES), default="nrftw",
        help="select the game configuration; the perception/generation loop remains shared",
    )
    parser.add_argument("--image", required=True, help="initial RGB frame")
    parser.add_argument("--player-prompts", required=True, help="dataset-format player prompt JSON/directory")
    parser.add_argument("--output-dir")
    parser.add_argument("--ticks", type=int, default=12, help="number of 5-second ticks")
    parser.add_argument(
        "--prompt-root",
        help="prompt vocabulary corpus (or set GAMEDIRECTOR_PROMPT_ROOT/per-game variable)",
    )
    parser.add_argument(
        "--vocabulary-file",
        help="compact bundled vocabulary snapshot; makes --prompt-root unnecessary",
    )
    parser.add_argument("--player-choice", type=int, default=1)
    parser.add_argument("--boss-choice", type=int, default=1)
    parser.add_argument("--background", help="NRFTW background suffix; auto-detected by default")
    parser.add_argument(
        "--strip-background", action="store_true",
        help="NRFTW only: strip background suffix for a legacy checkpoint",
    )

    parser.add_argument("--player-hp", type=int)
    parser.add_argument("--boss-hp", type=int)
    parser.add_argument("--player-attack", type=int)
    parser.add_argument("--boss-attack", type=int)
    parser.add_argument(
        "--boss-skill", type=int, default=None,
        help="initial boss blue meter; defaults to --boss-skill-max",
    )
    parser.add_argument(
        "--boss-skill-max", type=int, default=20,
        help="custom blue-meter capacity and full special-skill cost",
    )
    for skill in ALL_SKILL_NAMES:
        parser.add_argument(f"--cooldown-{skill.replace('_', '-')}", type=float, default=None)

    parser.add_argument("--attack-checkpoint", default=str(DEFAULT_ATTACK))
    parser.add_argument("--attack-device", default="auto")
    parser.add_argument("--attack-threshold", type=float)
    parser.add_argument("--situation-checkpoint", default=str(DEFAULT_SITUATION))
    parser.add_argument("--situation-device", default="auto")

    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8018/v1")
    parser.add_argument("--llm-model", default="gemma-4-E2B-it")
    parser.add_argument("--llm-checkpoint", default=str(DEFAULT_LLM_CHECKPOINT))
    parser.add_argument("--llm-server-python")
    parser.add_argument("--llm-device", default="cuda:0")
    parser.add_argument(
        "--llm-auto-start", action=argparse.BooleanOptionalAction, default=True,
        help="start the bundled local Gemma service when --llm-base-url is unavailable",
    )
    parser.add_argument("--llm-startup-timeout", type=float, default=300.0)
    parser.add_argument(
        "--allow-llm-fallback", action="store_true",
        help="explicitly allow heuristic planning when an LLM request or response fails",
    )
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-timeout", type=float, default=90.0)
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument("--plan-max", type=int, choices=(1, 2, 3), default=3)

    parser.add_argument(
        "--world-model-mode", choices=("ar", "bidirectional"), default="ar",
        help="causal per-cell AR closed loop (default) or legacy 5-second batched generation",
    )
    parser.add_argument(
        "--world-model-checkpoint",
        help="checkpoint file; defaults to the repository's mode-specific model/world_model entry",
    )
    parser.add_argument(
        "--base-model", default=str(DEFAULT_BASE_MODEL),
        help="directory containing Wan-AI assets (defaults to the repository model directory)",
    )
    parser.add_argument("--world-model-device", default="cuda")
    parser.add_argument("--world-model-steps", type=int, default=30)
    parser.add_argument("--world-model-cfg", type=float, default=2.0)
    parser.add_argument("--ar-stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--ar-sampler", choices=("euler", "distilled"), default="euler")
    parser.add_argument(
        "--ar-denoise-timesteps", default="1000,500,250",
        help="comma-separated timesteps used by distilled AR stages",
    )
    parser.add_argument(
        "--ar-attention-sink", action=argparse.BooleanOptionalAction, default=True,
        help="bound causal KV memory with three fixed and seventeen sliding latent frames",
    )
    parser.add_argument(
        "--ar-reanchor-between-ticks", action=argparse.BooleanOptionalAction, default=False,
        help="reset causal KV state and re-encode the RGB tail every five seconds",
    )
    parser.add_argument(
        "--ar-carry-clean-latent", action=argparse.BooleanOptionalAction, default=False,
        help="carry the last clean latent when re-anchoring instead of encoding RGB",
    )
    parser.add_argument(
        "--post-death-seconds", type=float, default=5.0,
        help=(
            "seconds of dying/no-action generation after death (default: 5); "
            "always capped by the requested tick budget"
        ),
    )
    parser.add_argument("--seed", type=int, default=2)
    return parser


def _selected_name(options, choice: int, label: str, game_name: str) -> str:
    try:
        return options[choice]
    except KeyError as error:
        rendered = ", ".join(f"{key}={value}" for key, value in options.items())
        raise SystemExit(f"invalid --{label}-choice for {game_name}; choose {rendered}") from error


def _auto_background(player_prompts: str, prompt_root: str | Path) -> str | None:
    candidate = Path(player_prompts).expanduser()
    if candidate.is_dir():
        candidate = next(iter(sorted(candidate.glob("clip_*.json"))), None)
    if candidate is None or not candidate.is_file():
        return None
    try:
        session_id = json.loads(candidate.read_text(encoding="utf-8")).get("session_id")
    except (json.JSONDecodeError, OSError):
        return None
    return detect_background(prompt_root, session_id) if session_id else None


def main() -> None:
    args = build_parser().parse_args()
    if args.ticks <= 0:
        raise SystemExit("--ticks must be positive")
    if args.post_death_seconds is not None and args.post_death_seconds < 0:
        raise SystemExit("--post-death-seconds must be non-negative")
    if args.boss_skill_max <= 0:
        raise SystemExit("--boss-skill-max must be positive")
    boss_skill = args.boss_skill_max if args.boss_skill is None else args.boss_skill
    if not 0 <= boss_skill <= args.boss_skill_max:
        raise SystemExit("--boss-skill must be between 0 and --boss-skill-max")
    for name in ("player_hp", "boss_hp", "player_attack", "boss_attack"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    llm_service = ensure_llm_server(
        base_url=args.llm_base_url,
        model=args.llm_model,
        checkpoint=args.llm_checkpoint,
        python=args.llm_server_python,
        device=args.llm_device,
        auto_start=args.llm_auto_start,
        startup_timeout=args.llm_startup_timeout,
    )
    if llm_service["started"]:
        print(
            f"Started bundled Gemma service (PID {llm_service['pid']}); "
            f"log: {llm_service['log']}",
            flush=True,
        )
    else:
        print(f"Using healthy Gemma service at {args.llm_base_url}", flush=True)
    game = get_game(args.game)
    player_name = _selected_name(game.players, args.player_choice, "player", game.key)
    boss_name = _selected_name(game.bosses, args.boss_choice, "boss", game.key)
    configured_prompt_root = None
    if not args.vocabulary_file:
        configured_prompt_root = (
            args.prompt_root
            or os.environ.get(f"GAMEDIRECTOR_{game.key.upper()}_PROMPT_ROOT")
            or os.environ.get("GAMEDIRECTOR_PROMPT_ROOT")
        )
        if not configured_prompt_root:
            raise SystemExit(
                "pass --vocabulary-file, --prompt-root, GAMEDIRECTOR_PROMPT_ROOT, or "
                f"GAMEDIRECTOR_{game.key.upper()}_PROMPT_ROOT"
            )
    prompt_root = Path(configured_prompt_root).expanduser() if configured_prompt_root else None
    background = args.background
    if game.key == "nrftw" and not args.strip_background and background is None:
        if prompt_root is not None:
            background = _auto_background(args.player_prompts, prompt_root)
        if background is None:
            raise SystemExit(
                "NRFTW prompts retain their arena suffix; pass --background or use a raw "
                "prompt corpus with a sidecar session_id so it can be auto-detected"
            )
    elif game.key != "nrftw" and (background or args.strip_background):
        raise SystemExit("--background/--strip-background apply only to nrftw")

    if args.vocabulary_file:
        if args.strip_background:
            raise SystemExit("--strip-background cannot be used with --vocabulary-file")
        vocabulary = PromptVocabulary.from_snapshot(
            args.vocabulary_file,
            game=game,
            player_name=player_name,
            boss_name=boss_name,
            background=background,
        )
    else:
        vocabulary = PromptVocabulary(
            prompt_root,
            game=game,
            player_name=player_name,
            boss_name=boss_name,
            background=background,
            strip_background=args.strip_background,
        )
    player_track = PlayerPromptTrack.from_path(args.player_prompts, vocabulary)
    cooldown_overrides = {
        skill: getattr(args, f"cooldown_{skill}")
        for skill in ALL_SKILL_NAMES
        if skill in game.skill_library[boss_name]
        and getattr(args, f"cooldown_{skill}") is not None
    }
    planner_kwargs = {
        "base_url": args.llm_base_url,
        "model": args.llm_model,
        "temperature": args.llm_temperature,
        "timeout": args.llm_timeout,
        "retries": args.llm_retries,
        "allow_fallback": args.allow_llm_fallback,
    }
    cards, planner, cooldowns, filler_track = make_runtime_decision_components(
        game=game,
        boss_name=boss_name,
        timelines=vocabulary.timelines,
        seed=args.seed,
        cooldown_overrides=cooldown_overrides,
        planner_kwargs=planner_kwargs,
        plan_max=args.plan_max,
        skill_cost=args.boss_skill_max,
    )

    # These are always the real three-game models; there is no mock branch.
    attacknet = AttackNetRunner(
        args.attack_checkpoint, device=args.attack_device, threshold=args.attack_threshold
    )
    situationnet = SituationNetRunner(
        args.situation_checkpoint,
        device=args.situation_device,
        distance_unit=game.distance_unit,
    )
    if args.world_model_mode == "ar":
        checkpoint = args.world_model_checkpoint or str(DEFAULT_AR_WORLD_MODEL)
        try:
            denoise_timesteps = tuple(
                int(value.strip()) for value in args.ar_denoise_timesteps.split(",") if value.strip()
            )
        except ValueError as error:
            raise SystemExit("--ar-denoise-timesteps must be comma-separated integers") from error
        if not denoise_timesteps:
            raise SystemExit("--ar-denoise-timesteps must not be empty")
        world_model = ARWorldModelRunner(
            checkpoint=checkpoint,
            base_model=args.base_model,
            stage=args.ar_stage,
            sampler=args.ar_sampler,
            device=args.world_model_device,
            steps=args.world_model_steps,
            denoise_timesteps=denoise_timesteps,
            cfg=args.world_model_cfg,
            height=480,
            width=832,
            streaming_vae_decode=True,
            carry_clean_latent_between_ticks=args.ar_carry_clean_latent,
            attention_sink=args.ar_attention_sink,
            reanchor_between_ticks=args.ar_reanchor_between_ticks,
        )
    else:
        checkpoint = args.world_model_checkpoint or str(DEFAULT_WORLD_MODEL)
        world_model = WorldModelRunner(
            checkpoint=checkpoint,
            base_model=args.base_model,
            device=args.world_model_device,
            steps=args.world_model_steps,
            cfg=args.world_model_cfg,
        )

    def value_or_default(value, default):
        return default if value is None else value

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args)
    common = {
        "game": game,
        "vocabulary": vocabulary,
        "player_track": player_track,
        "planner": planner,
        "attacknet": attacknet,
        "situationnet": situationnet,
        "world_model": world_model,
        "output_dir": output_dir,
        "cards": cards,
        "cooldowns": cooldowns,
        "filler_track": filler_track,
        "state": CombatState(
            player_hp=value_or_default(args.player_hp, game.default_player_hp),
            boss_hp=value_or_default(args.boss_hp, game.default_boss_hp),
            player_attack=value_or_default(args.player_attack, game.default_player_attack),
            boss_attack=value_or_default(args.boss_attack, game.default_boss_attack),
            boss_skill=boss_skill,
            boss_skill_max=args.boss_skill_max,
        ),
        "seed": args.seed,
        "plan_max": args.plan_max,
    }
    if args.world_model_mode == "ar":
        agent = ClosedLoopAgentAR(
            **common,
            post_death_seconds=args.post_death_seconds,
        )
    else:
        agent = ClosedLoopAgent(**common)
    summary = agent.run(args.image, ticks=args.ticks)
    print(
        f"Completed {summary['duration_s']:.2f}s of {game.display_name}: "
        f"{player_name} vs {boss_name}; final combat={summary['final_combat']}",
        flush=True,
    )
    print(f"Outputs: {output_dir.expanduser().resolve()}", flush=True)


if __name__ == "__main__":
    main()
