#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from flsim.utils.config_utils import fullclassname
from flsim.clients.base_client import Client, ClientConfig


class TieredClient(Client):
    r"""
    Class to represent a single tiered client.
    """

    def get_total_training_time(self) -> float:
        return self.dataset.user_tier() * self.timeout_simulator.simulate_training_time(
            self.per_example_training_time, self.dataset.num_train_examples()
        )

@dataclass
class TieredClientConfig(ClientConfig):
    _target_: str = fullclassname(TieredClient)