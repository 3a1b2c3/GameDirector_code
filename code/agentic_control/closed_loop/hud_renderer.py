"""Render the combat HUD (health bars, boss info card, key overlay) over rollout frames."""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .components import _rgb_array


_KEY_SIZE = 22
_KEY_GAP = 4
_ACTIVE_COLOR = (60, 200, 60)  # BGR: green
_IDLE_COLOR = (70, 70, 70)
_BORDER_COLOR = (220, 220, 220)
_FONT_REGULAR = Path("/usr/share/fonts/truetype/lato/Lato-Regular.ttf")
_FONT_BOLD = Path("/usr/share/fonts/truetype/lato/Lato-Bold.ttf")


@lru_cache(maxsize=32)
def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = _FONT_BOLD if bold else _FONT_REGULAR
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _rgba(rgb: tuple[int, int, int], alpha: int = 255) -> tuple[int, int, int, int]:
    return (*rgb, alpha)


def _humanize(value: str, *, movement: bool = False) -> str:
    value = str(value or "none").strip().lower()
    if value in {"none", "no_action", "no action"}:
        return "Stationary" if movement else "None"
    label = value.replace("_", " ").replace("-", " ").title()
    return f"Move {label}" if movement else label


def _rounded_panel(
    layer: Image.Image,
    box: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
    width: int = 1,
    radius: int = 10,
) -> None:
    ImageDraw.Draw(layer).rounded_rectangle(
        box, radius=radius, fill=fill, outline=outline, width=width
    )


def _draw_info_card(
    layer: Image.Image,
    box: tuple[int, int, int, int],
    label: str,
    value: str,
    accent: tuple[int, int, int],
    *,
    scale: float,
) -> None:
    """Draw one HUD info card."""
    draw = ImageDraw.Draw(layer)
    x1, y1, x2, y2 = box
    _rounded_panel(
        layer,
        box,
        fill=_rgba((8, 12, 19), 195),
        outline=_rgba(accent, 205),
        width=max(1, round(2 * scale)),
        radius=max(8, round(11 * scale)),
    )
    bar_width = max(4, round(5 * scale))
    draw.rounded_rectangle(
        (
            x1 + round(7 * scale),
            y1 + round(8 * scale),
            x1 + round(7 * scale) + bar_width,
            y2 - round(8 * scale),
        ),
        radius=bar_width // 2,
        fill=_rgba(accent),
    )
    text_x = x1 + round(23 * scale)
    draw.text(
        (text_x, y1 + round(8 * scale)),
        label,
        font=_font(max(9, round(10 * scale)), bold=True),
        fill=_rgba(accent),
    )
    draw.text(
        (text_x, y1 + round(25 * scale)),
        value,
        font=_font(max(14, round(17 * scale)), bold=True),
        fill=_rgba((255, 255, 255)),
    )


def _draw_boss_info(
    frame: Image.Image,
    *,
    boss_name: str,
    boss_action: str,
    boss_movement: str,
) -> Image.Image:
    """Add the live Boss skill card without covering the existing bars."""
    base = frame.convert("RGB")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    scale = min(base.width / 832.0, base.height / 480.0)
    margin = max(10, round(16 * scale))
    panel_width = max(228, round(258 * scale))
    card_height = max(52, round(61 * scale))
    x2 = layer.width - margin
    x1 = x2 - panel_width

    # Keep the single skill card in the bottom-right, clear of the health and
    # skill-resource bars at the top of the frame.
    y1 = layer.height - margin - card_height
    draw.text(
        (x2, y1 - round(25 * scale)),
        str(boss_name or "Boss").upper(),
        anchor="ra",
        font=_font(max(11, round(13 * scale)), bold=True),
        fill=_rgba((237, 242, 248), 245),
        stroke_width=max(1, round(scale)),
        stroke_fill=_rgba((0, 0, 0), 175),
    )
    _draw_info_card(
        layer,
        (x1, y1, x2, y1 + card_height),
        "CURRENT SKILL",
        _humanize(boss_action),
        (255, 190, 74),
        scale=scale,
    )
    return Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")


def _draw_key(bgr: np.ndarray, x: int, y: int, label: str, active: bool) -> None:
    import cv2

    color = _ACTIVE_COLOR if active else _IDLE_COLOR
    cv2.rectangle(bgr, (x, y), (x + _KEY_SIZE, y + _KEY_SIZE), color, -1)
    cv2.rectangle(bgr, (x, y), (x + _KEY_SIZE, y + _KEY_SIZE), _BORDER_COLOR, 1)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
    tx = x + (_KEY_SIZE - text_size[0]) // 2
    ty = y + (_KEY_SIZE + text_size[1]) // 2
    cv2.putText(bgr, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_wasd_and_mouse(
    bgr: np.ndarray, *, player_action: str, player_movement: str
) -> None:
    """Bottom-left WASD + left-mouse-button indicator for the live player prompt."""
    import cv2

    height, _ = bgr.shape[:2]
    margin = 16
    row1_y = height - margin - (2 * _KEY_SIZE + _KEY_GAP)
    row2_y = row1_y + _KEY_SIZE + _KEY_GAP
    col_a = margin
    col_s = col_a + _KEY_SIZE + _KEY_GAP
    col_d = col_s + _KEY_SIZE + _KEY_GAP

    directions = set(str(player_movement).split("_"))
    _draw_key(bgr, col_s, row1_y, "W", "up" in directions)
    _draw_key(bgr, col_a, row2_y, "A", "left" in directions)
    _draw_key(bgr, col_s, row2_y, "S", "down" in directions)
    _draw_key(bgr, col_d, row2_y, "D", "right" in directions)

    mouse_x0 = col_d + _KEY_SIZE + 3 * _KEY_GAP
    mouse_y0 = row1_y
    mouse_w, mouse_h = 30, 2 * _KEY_SIZE + _KEY_GAP
    mouse_x1, mouse_y1 = mouse_x0 + mouse_w, mouse_y0 + mouse_h
    mid_x = (mouse_x0 + mouse_x1) // 2
    button_split_y = mouse_y0 + int(mouse_h * 0.43)

    attacking = player_action == "melee_attack"
    # A pill-shaped body reads as a mouse at HUD scale much more clearly than
    # the old rectangular indicator.  Build a mask so the active left button
    # follows the curved shell instead of spilling outside it.
    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    center = (mid_x, (mouse_y0 + mouse_y1) // 2)
    axes = (mouse_w // 2, mouse_h // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
    bgr[mask > 0] = _IDLE_COLOR
    if attacking:
        active_mask = mask.copy()
        active_mask[:, mid_x + 1 :] = 0
        active_mask[button_split_y + 1 :, :] = 0
        bgr[active_mask > 0] = _ACTIVE_COLOR

    cv2.ellipse(bgr, center, axes, 0, 0, 360, _BORDER_COLOR, 1, cv2.LINE_AA)
    half_width = int(axes[0] * 0.9)
    cv2.line(
        bgr,
        (mid_x - half_width, button_split_y),
        (mid_x + half_width, button_split_y),
        _BORDER_COLOR,
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        bgr,
        (mid_x, mouse_y0 + 2),
        (mid_x, button_split_y),
        _BORDER_COLOR,
        1,
        cv2.LINE_AA,
    )
    wheel_top = mouse_y0 + 7
    wheel_bottom = wheel_top + 9
    cv2.rectangle(
        bgr,
        (mid_x - 2, wheel_top),
        (mid_x + 2, wheel_bottom),
        (28, 28, 28),
        -1,
    )
    cv2.rectangle(
        bgr,
        (mid_x - 2, wheel_top),
        (mid_x + 2, wheel_bottom),
        _BORDER_COLOR,
        1,
    )


def _draw_outcome_overlay(
    frame: Image.Image,
    *,
    outcome: str,
    frame_index: int,
    fade_seconds: float,
    fps: int,
) -> Image.Image:
    """Fade and gently scale a centered You Win/You Lose result into view."""
    base = frame.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    fade_frames = max(1, round(max(0.0, fade_seconds) * fps))
    progress = min(1.0, max(0.0, (frame_index + 1) / fade_frames))
    eased = progress * progress * (3.0 - 2.0 * progress)
    scale = min(base.width / 832.0, base.height / 480.0)
    font_size = max(28, round((58.0 + 14.0 * eased) * scale))
    font = _font(font_size, bold=True)
    text = str(outcome)
    box = draw.textbbox((0, 0), text, font=font, stroke_width=max(1, round(2 * scale)))
    text_width = box[2] - box[0]
    text_height = box[3] - box[1]
    center_x, center_y = base.width // 2, base.height // 2
    pad_x, pad_y = round(34 * scale), round(20 * scale)
    panel = (
        center_x - text_width // 2 - pad_x,
        center_y - text_height // 2 - pad_y,
        center_x + text_width // 2 + pad_x,
        center_y + text_height // 2 + pad_y,
    )
    _rounded_panel(
        layer,
        panel,
        fill=(3, 6, 11, round(150 * eased)),
        outline=(245, 245, 245, round(95 * eased)),
        width=max(1, round(2 * scale)),
        radius=max(12, round(18 * scale)),
    )
    color = (255, 92, 92) if text == "You Lose" else (255, 214, 82)
    position = (center_x, center_y)
    draw.text(
        (center_x + round(3 * scale), center_y + round(4 * scale)),
        text,
        anchor="mm",
        font=font,
        fill=(0, 0, 0, round(190 * eased)),
        stroke_width=max(2, round(4 * scale)),
        stroke_fill=(0, 0, 0, round(150 * eased)),
    )
    draw.text(
        position,
        text,
        anchor="mm",
        font=font,
        fill=(*color, round(255 * eased)),
        stroke_width=max(1, round(2 * scale)),
        stroke_fill=(255, 255, 255, round(135 * eased)),
    )
    return Image.alpha_composite(base, layer).convert("RGB")


def draw_health_hud_cell(
    frames: Sequence[Image.Image | np.ndarray],
    state: dict[str, Any],
    *,
    player_max_hp: int,
    boss_max_hp: int,
) -> list[Image.Image]:
    """Draw one fixed HP snapshot onto exactly 4 frames (one prompt cell) of the
    closed loop, which generates and HP-checks 4 frames at a time. Frames fed
    back into the world model are never modified."""
    import cv2

    if len(frames) != 4:
        raise ValueError(f"AR health HUD requires exactly 4 frames, got {len(frames)}")
    output = []
    for frame in frames:
        bgr = cv2.cvtColor(_rgb_array(frame).copy(), cv2.COLOR_RGB2BGR)
        _, width = bgr.shape[:2]
        bar_width = min(280, max(160, width // 3))
        bar_height, margin, top = 22, 24, 26

        def bar(x: int, hp: int, maximum: int, label: str, align_right: bool) -> None:
            ratio = max(0.0, min(1.0, hp / maximum))
            cv2.rectangle(
                bgr, (x - 8, top - 22), (x + bar_width + 8, top + bar_height + 8),
                (10, 10, 10), -1,
            )
            cv2.rectangle(bgr, (x, top), (x + bar_width, top + bar_height), (35, 35, 35), -1)
            fill = int(round(bar_width * ratio))
            start = x + bar_width - fill if align_right else x
            color = (55, 55, 230) if ratio <= 0.25 else (55, 185, 55)
            cv2.rectangle(bgr, (start, top), (start + fill, top + bar_height), color, -1)
            cv2.rectangle(bgr, (x, top), (x + bar_width, top + bar_height), (235, 235, 235), 1)
            text = f"{label}  {hp} / {maximum}"
            text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
            text_x = x + bar_width - text_width if align_right else x
            cv2.putText(
                bgr, text, (text_x, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

        bar(margin, int(state["player_hp"]), player_max_hp, "PLAYER", False)
        bar(width - margin - bar_width, int(state["boss_hp"]), boss_max_hp, "BOSS", True)
        output.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    return output


def concat_mp4(clips: Sequence[str | Path], output: str | Path) -> None:
    """Losslessly concatenate same-format five-second clips."""
    paths = [Path(path).resolve() for path in clips]
    if not paths:
        raise ValueError("cannot concatenate an empty clip list")
    destination = Path(output).resolve()
    if len(paths) == 1:
        shutil.copyfile(paths[0], destination)
        return
    manifest = destination.with_suffix(".concat.txt")
    with manifest.open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"file '{path}'\n")
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
             "-safe", "0", "-i", str(manifest), "-c", "copy", str(destination)],
            check=True,
        )
    finally:
        manifest.unlink(missing_ok=True)


def draw_combat_hud_cell(
    frames: Sequence[Image.Image | np.ndarray],
    state: dict[str, Any],
    *,
    player_max_hp: int,
    boss_max_hp: int,
    player_action: str,
    player_movement: str,
    boss_name: str,
    boss_action: str,
    boss_movement: str,
    outcome: str | None = None,
    outcome_frame_start: int = 0,
    outcome_fade_seconds: float = 2.0,
    fps: int = 16,
) -> list[Image.Image]:
    """Draw combat HUDs and an optional animated centered match outcome."""
    import cv2

    health_frames = draw_health_hud_cell(
        frames,
        state,
        player_max_hp=player_max_hp,
        boss_max_hp=boss_max_hp,
    )
    output = []
    for frame_offset, frame in enumerate(health_frames):
        rgb = np.asarray(frame.convert("RGB"), dtype=np.uint8)
        bgr = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
        _, width = bgr.shape[:2]
        bar_width = min(280, max(160, width // 3))
        bar_height, margin, top = 10, 24, 62
        x = width - margin - bar_width
        current = int(state["boss_skill"])
        maximum = int(state["boss_skill_max"])
        ratio = max(0.0, min(1.0, current / max(maximum, 1)))
        cv2.rectangle(bgr, (x, top), (x + bar_width, top + bar_height), (35, 35, 35), -1)
        fill = int(round(bar_width * ratio))
        cv2.rectangle(bgr, (x + bar_width - fill, top), (x + bar_width, top + bar_height), (255, 130, 30), -1)
        cv2.rectangle(bgr, (x, top), (x + bar_width, top + bar_height), (235, 235, 235), 1)
        label = f"SKILL {current} / {maximum}"
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
        cv2.putText(
            bgr,
            label,
            (x + bar_width - size, top + bar_height + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        _draw_wasd_and_mouse(bgr, player_action=player_action, player_movement=player_movement)
        rendered = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        rendered = _draw_boss_info(
            rendered,
            boss_name=boss_name,
            boss_action=boss_action,
            boss_movement=boss_movement,
        )
        if outcome:
            rendered = _draw_outcome_overlay(
                rendered,
                outcome=outcome,
                frame_index=outcome_frame_start + frame_offset,
                fade_seconds=outcome_fade_seconds,
                fps=fps,
            )
        output.append(rendered)
    return output


__all__ = ["concat_mp4", "draw_combat_hud_cell"]