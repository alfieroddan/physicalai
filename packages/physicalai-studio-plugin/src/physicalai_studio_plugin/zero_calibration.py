# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Guided zero-pose calibration that a plugin can offer for its robot types."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from physicalai.robot.interface import Robot as PhysicalAIRobot


@dataclass(frozen=True)
class RobotZeroCalibration:
    """Zero-pose calibration steps for a robot type.

    Studio runs these on its own exclusive connection to the plain driver
    returned by the catalog builder, before the robot is added. It shows
    ``instructions`` next to a live 3D view, calls ``release`` so the arm can
    be moved by hand, calls ``set_zero`` when the user confirms the pose, and
    then checks that every joint reads within ``zero_tolerance_deg`` of zero.

    Attributes:
        instructions: Plain-text description of the zero pose shown to the user.
        set_zero: Stores the arm's current pose as zero on its motors.
        release: Optional step that makes the connected arm movable by hand,
            for example by disabling torque. ``None`` when the arm is already
            passive after ``connect()``.
        zero_tolerance_deg: Largest absolute joint reading, in degrees, that
            counts as zero when Studio verifies the result.
    """

    instructions: str
    set_zero: Callable[[PhysicalAIRobot], Awaitable[None]]
    release: Callable[[PhysicalAIRobot], Awaitable[None]] | None = None
    zero_tolerance_deg: float = 5.0

    def __post_init__(self) -> None:
        """Validate the calibration definition.

        Raises:
            ValueError: If ``instructions`` is blank or ``zero_tolerance_deg`` is not a finite positive value.
        """
        if not self.instructions.strip():
            msg = "instructions must not be empty"
            raise ValueError(msg)
        if not math.isfinite(self.zero_tolerance_deg) or self.zero_tolerance_deg <= 0.0:
            msg = f"zero_tolerance_deg must be a finite positive value, got {self.zero_tolerance_deg!r}"
            raise ValueError(msg)
