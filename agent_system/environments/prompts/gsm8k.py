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

GSM8K_TEMPLATE_NO_HIS = """
You are a math expert. You are given a question and you need to solve it step by step.
Your question: {task_description}

Now it's your turn to solve the problem.
You should reason step by step before any tool call.
You should use the `calc_gsm8k_reward` tool after step by step solving the question, 
before generate final answer at least once and refine your answer if necessary. 
Put your final answer in the format of `#### <answer>`.
"""

GSM8K_TEMPLATE = """
You are a math expert. You are given a question and you need to solve it step by step.
Your question: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below is the interaction history:
{memory_context}

Now it's your turn to solve the problem.
You should reason step by step before any tool call.
You should use the `calc_gsm8k_reward` tool after step by step solving the question, 
before generate final answer at least once and refine your answer if necessary. 
Put your final answer in the format of `#### <answer>`.
"""

