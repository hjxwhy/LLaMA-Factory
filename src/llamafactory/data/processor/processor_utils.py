# Copyright 2025 the LlamaFactory team.
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

import bisect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..template import Template


@dataclass
class DatasetProcessor(ABC):
    r"""A class for data processors."""

    template: "Template"
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]
    data_args: "DataArguments"

    @abstractmethod
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        r"""Build model inputs from the examples."""
        ...

    @abstractmethod
    def print_data_example(self, example: dict[str, list[int]]) -> None:
        r"""Print a data example to stdout."""
        ...


def search_for_fit(numbers: list[int], capacity: int) -> int:
    r"""Find the index of largest number that fits into the knapsack with the given capacity."""
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


def greedy_knapsack(numbers: list[int], capacity: int) -> list[list[int]]:
    r"""Implement efficient greedy algorithm with binary search for the knapsack problem."""
    numbers.sort()  # sort numbers in ascending order for binary search
    knapsacks = []

    while numbers:
        current_knapsack = []
        remaining_capacity = capacity

        while True:
            index = search_for_fit(numbers, remaining_capacity)
            if index == -1:
                break  # no more numbers fit in this knapsack

            remaining_capacity -= numbers[index]  # update the remaining capacity
            current_knapsack.append(numbers.pop(index))  # add the number to knapsack

        knapsacks.append(current_knapsack)

    return knapsacks

def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    r"""
    Compute the real sequence length after truncation by the cutoff_len.
    
    这个函数用于计算在给定最大长度限制(cutoff_len)下，如何分配source和target的长度。
    
    逻辑说明：
    1. 如果target长度的2倍小于cutoff_len，说明target相对较短，优先保留完整的target，
       允许其使用完整的cutoff_len空间，主要截断source部分
    2. 如果source长度的2倍小于cutoff_len，说明source相对较短，优先保留完整的source，
       target最多只能使用剩余空间(cutoff_len - source_len)
    3. 如果source和target都比较长，则按比例分配空间：
       target分配到的最大长度 = cutoff_len * (target_len / (source_len + target_len))
       这样可以保持原有的source:target长度比例
    
    最后确保：
    - new_target_len不超过原始target_len和分配的max_target_len
    - new_source_len不超过原始source_len和剩余空间max_source_len
    """
    if target_len * 2 < cutoff_len:  # target相对较短，优先保留target，主要截断source
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # source相对较短，优先保留source，截断target
        max_target_len = cutoff_len - source_len
    else:  # 两者都较长，按比例分配空间
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    # 计算实际的新长度
    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len


def infer_seqlen_v2(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    r"""
    Compute the real sequence length after truncation by the cutoff_len.
    优先保留source长度的版本。
    
    这个函数用于计算在给定最大长度限制(cutoff_len)下，如何分配source和target的长度。
    优先保留source长度。
    
    逻辑说明：
    1. 首先尝试保留完整的source长度
    2. 如果source长度不超过cutoff_len，则target可以使用剩余空间
    3. 如果source长度超过cutoff_len，则截断source到cutoff_len，target长度为0
    4. 确保总长度不超过cutoff_len
    """
    # 优先保留source长度
    new_source_len = min(source_len, cutoff_len)
    
    # target使用剩余空间
    remaining_capacity = cutoff_len - new_source_len
    new_target_len = min(target_len, remaining_capacity)
    
    return new_source_len, new_target_len
