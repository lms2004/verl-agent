# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import concurrent.futures
from typing import Any, Dict, List, Tuple

import gym
import numpy as np
from omegaconf import DictConfig


class GSM8KMultiProcessEnv(gym.Env):
    """
    GSM8K environment for multi-turn tool calling.
    - env_num  : Number of groups (logical sharding; keep the parameter for external compatibility)
    - group_n  : Number of environments per group
    - total_envs = env_num * group_n
    
    This environment is designed for dataset-based training where tasks come from a dataset.
    The environment manages the state of each problem and tracks tool calls.
    """

    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
        env_config: DictConfig | None = None,
    ) -> None:
        super().__init__()

        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.is_train = is_train
        self.max_steps = env_config.max_steps if env_config else 10

        self._rng = np.random.RandomState(seed)

        # Store environment states: each env has a question, ground_truth, and step count
        self.env_states = [None] * self.batch_size

        max_workers = min(self.batch_size, 256)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    def _sync_reset(self, env_idx: int, kwargs: Dict):
        """Reset a single environment with the given kwargs."""
        question = kwargs.get("question", "")
        ground_truth = kwargs.get("ground_truth", "")
        data_source = kwargs.get("data_source", "unknown")
        
        # Store state for this environment
        self.env_states[env_idx] = {
            "question": question,
            "ground_truth": ground_truth,
            "data_source": data_source,
            "step_count": 0,
            "done": False,
        }
        
        # Return the question as observation
        obs = question
        info = {
            "data_source": data_source,
            "ground_truth": ground_truth,
        }
        return obs, info

    def _sync_step(self, env_idx: int, action: str):
        """Step a single environment with the given action."""
        state = self.env_states[env_idx]
        if state is None:
            # Environment not initialized
            return "", 0.0, True, {"error": "Environment not initialized"}
        
        state["step_count"] += 1
        
        # For GSM8K, the reward is computed by the reward model, not the environment
        # The environment just tracks the state and returns done when max_steps is reached
        reward = 0.0  # Reward will be computed by reward_fn later
        done = state["step_count"] >= self.max_steps or state.get("done", False)
        
        # Return empty observation (the actual observation comes from the model's response)
        # For multi-turn tool calling, the observation is the tool response
        obs = ""  # Tool responses are handled by sglang engine
        
        info = {
            "data_source": state["data_source"],
            "ground_truth": state["ground_truth"],
            "step_count": state["step_count"],
            "question": state["question"],
        }
        
        return obs, reward, done, info

    def reset(self, kwargs: List[Dict]):
        """Reset all environments with the given kwargs."""
        if len(kwargs) > self.batch_size:
            raise ValueError(
                f"Got {len(kwargs)} kwarg dicts, but the env was initialised with total_envs={self.batch_size}"
            )

        pad_n = self.batch_size - len(kwargs)
        dummy_kw = {
            "ground_truth": "",
            "question": "",
            "data_source": "unknown",
        }

        padded_kwargs = list(kwargs) + [dummy_kw] * pad_n
        valid_mask = [True] * len(kwargs) + [False] * pad_n

        tasks = [
            self._loop.run_in_executor(
                self._executor, self._sync_reset, idx, kw
            )
            for idx, kw in enumerate(padded_kwargs)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))

        obs_list, info_list = map(list, zip(*results))

        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        info_list = [i for i, keep in zip(info_list, valid_mask) if keep]

        return obs_list, info_list

    def step(self, actions: List[str]):
        """Step all environments with the given actions."""
        if len(actions) > self.batch_size:
            raise ValueError(
                f"Got {len(actions)} actions, but the env was initialized with total_envs={self.batch_size}"
            )

        pad_n = self.batch_size - len(actions)
        padded_actions = list(actions) + [""] * pad_n
        valid_mask = [True] * len(actions) + [False] * pad_n

        tasks = [
            self._loop.run_in_executor(
                self._executor, self._sync_step, idx, act
            )
            for idx, act in enumerate(padded_actions)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))

        obs_list, reward_list, done_list, info_list = map(list, zip(*results))

        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        reward_list = [r for r, keep in zip(reward_list, valid_mask) if keep]
        done_list = [d for d, keep in zip(done_list, valid_mask) if keep]
        info_list = [i for i, keep in zip(info_list, valid_mask) if keep]

        return obs_list, reward_list, done_list, info_list

    def close(self):
        """Close all environments."""
        if getattr(self, "_closed", False):
            return
        self._executor.shutdown(wait=True)
        self._loop.close()
        self._closed = True
        self.env_states = [None] * self.batch_size

    def __del__(self):
        self.close()


def build_gsm8k_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_config=None,
):
    """Build GSM8K environments."""
    return GSM8KMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        env_config=env_config,
    )

