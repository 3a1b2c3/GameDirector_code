"""Combat HP and boss-skill-resource bookkeeping shared by all three games."""

from __future__ import annotations

from dataclasses import asdict, dataclass


SKILL_BAR_MAX = 20


@dataclass
class CombatState:
    player_hp: int = 1000
    boss_hp: int = 1000
    player_attack: int = 40
    boss_attack: int = 20
    boss_skill: int = SKILL_BAR_MAX
    boss_skill_max: int = SKILL_BAR_MAX

    def __post_init__(self) -> None:
        for name in (
            "player_hp",
            "boss_hp",
            "player_attack",
            "boss_attack",
            "boss_skill",
            "boss_skill_max",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if self.boss_skill_max <= 0:
            raise ValueError("boss_skill_max must be positive")
        if self.boss_skill > self.boss_skill_max:
            raise ValueError("boss_skill cannot exceed boss_skill_max")

    @property
    def player_dead(self) -> bool:
        return self.player_hp <= 0

    @property
    def boss_dead(self) -> bool:
        return self.boss_hp <= 0

    @property
    def skill_ready(self) -> bool:
        return self.boss_skill >= self.boss_skill_max

    def spend_skill(self) -> None:
        """Spend the full user-configured bar for one special-skill release."""
        if not self.skill_ready:
            raise ValueError(
                f"boss skill must be full ({self.boss_skill_max}/{self.boss_skill_max}), "
                f"got {self.boss_skill}/{self.boss_skill_max}"
            )
        self.boss_skill = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class DamageCalculator:
    """Gate AttackNet detections with attack prompts, then apply damage.

    A detection only causes damage when the opposing actor's emitted prompt
    is an attack. Each actor that actually loses HP adds ``skill_gain_per_hit``
    points to the shared Boss skill bar.
    """

    @staticmethod
    def apply_cell(
        state: CombatState,
        *,
        player_hit: bool,
        boss_hit: bool,
        player_attack_prompt: bool,
        boss_attack_prompt: bool,
        skill_gain_per_hit: int = 1,
    ) -> dict[str, object]:
        if skill_gain_per_hit <= 0:
            raise ValueError("skill_gain_per_hit must be positive")
        before = state.as_dict()
        player_was_dead, boss_was_dead = state.player_dead, state.boss_dead
        player_hit_detected = bool(player_hit)
        boss_hit_detected = bool(boss_hit)
        player_can_take_damage = bool(
            player_hit_detected
            and boss_attack_prompt
            and not player_was_dead
            and not boss_was_dead
        )
        boss_can_take_damage = bool(
            boss_hit_detected
            and player_attack_prompt
            and not boss_was_dead
            and not player_was_dead
        )

        if player_can_take_damage:
            state.player_hp = max(0, state.player_hp - state.boss_attack)
        if boss_can_take_damage:
            state.boss_hp = max(0, state.boss_hp - state.player_attack)

        player_damage = before["player_hp"] - state.player_hp
        boss_damage = before["boss_hp"] - state.boss_hp
        landed_player_hit = player_damage > 0
        landed_boss_hit = boss_damage > 0
        gain = int(skill_gain_per_hit) * (
            int(landed_player_hit) + int(landed_boss_hit)
        )
        state.boss_skill = min(state.boss_skill_max, state.boss_skill + gain)
        return {
            "player_hit": landed_player_hit,
            "boss_hit": landed_boss_hit,
            "player_hit_detected": player_hit_detected,
            "boss_hit_detected": boss_hit_detected,
            "player_attack_prompt": bool(player_attack_prompt),
            "boss_attack_prompt": bool(boss_attack_prompt),
            "player_damage": player_damage,
            "boss_damage": boss_damage,
            "skill_gain": gain,
            "before": before,
            "after": state.as_dict(),
        }


__all__ = ["SKILL_BAR_MAX", "CombatState", "DamageCalculator"]
