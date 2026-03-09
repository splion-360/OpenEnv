# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

try:
    from .models import RewardBreakdown
except ImportError:
    try:
        from models import RewardBreakdown  # type: ignore
    except ImportError:
        from pick_and_place_env.models import RewardBreakdown  # type: ignore


def compute_reward_breakdown(success: bool) -> RewardBreakdown:
    reward = 1.0 if success else 0.0
    return RewardBreakdown(success=reward, total=reward)
