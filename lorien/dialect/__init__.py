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
Load Dialect Modules
"""
from typing import Set
import importlib

AVAILABLE_DIALECTS: Set[str] = set()

if importlib.util.find_spec("tvm") is not None:
    import tvm

    from . import tvm_dial
    from .tvm_dial.util import check_tvm_version

    # Minimum supported version of TVM.
    TVM_MIN_VERSION = "0.8.dev0"

    # Check if the TVM version is supported.
    TVM_VERSION = tvm.__version__
    if not check_tvm_version(TVM_VERSION, TVM_MIN_VERSION):
        raise RuntimeError("Unsatisfied TVM version (>= %s): %s" % (TVM_MIN_VERSION, TVM_VERSION))

    AVAILABLE_DIALECTS.add("tvm")
    AVAILABLE_DIALECTS.add("autotvm")
    AVAILABLE_DIALECTS.add("auto_scheduler")
