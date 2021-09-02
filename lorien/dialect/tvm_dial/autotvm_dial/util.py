# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utility functions for AutoTVM dialects
"""
from typing import Callable, Optional, Tuple

from tvm.autotvm.graph_tuner.base_graph_tuner import get_infer_layout
from tvm.autotvm.measure import MeasureInput, MeasureResult


def infer_task_layout(record: Tuple[MeasureInput, MeasureResult]) -> Optional[Tuple]:
    """Infer the layout of the given task. Return None if the layout cannot be inferred.

    Parameters
    ----------
    record: Tuple[MeasureInput, MeasureResult]
        The AutoTVM record pair.

    Return
    ------
    layout: Optional[Tuple]
        A tuple of input and output layout, or None if not inferrable.
    """
    infer_layout_func: Optional[Callable] = None
    try:
        infer_layout_func = get_infer_layout(record[0].task.name)
        assert infer_layout_func is not None
        with record[0].target:
            return infer_layout_func(record[0].task.workload, record[0].config)
    except ValueError:
        pass
    return None
