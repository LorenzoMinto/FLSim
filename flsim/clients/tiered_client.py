#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional, Tuple

from flsim.clients.base_client import Client, ClientConfig


class TieredClient(Client):
    r"""
    Class to represent a single tiered client.
    """

    def get_total_training_time(self) -> float:
        return self.dataset.user_tier() * super.get_total_training_time()

@dataclass
class TieredClientConfig(ClientConfig):
    _target_: str = fullclassname(TieredClient)