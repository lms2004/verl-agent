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

from typing import List, Tuple


def gsm8k_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """Project a list of LLM actions for GSM8K environment.
    
    For GSM8K, the actions are primarily tool calls and final answers.
    Since tool calling is handled by sglang engine, we mainly validate
    that the action is not empty.
    
    Args:
        actions: List of action strings from the model
        
    Returns:
        Tuple of (projected_actions, valids):
        - projected_actions: The actions (unchanged for GSM8K)
        - valids: List of 1/0 indicating if each action is valid
    """
    results: List[str] = []
    valids: List[int] = []
    
    for action in actions:
        # For GSM8K, we accept any non-empty action
        # Tool calling validation is handled by sglang engine
        if action and action.strip():
            results.append(action)
            valids.append(1)
        else:
            results.append("")
            valids.append(0)
    
    return results, valids

