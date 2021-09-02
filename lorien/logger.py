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
The format and config of logging.
"""

import logging
from typing import Callable, Dict

from .util import get_time_str

LOGGER_TABLE: Dict[str, logging.Logger] = {}

FORMATTER = logging.Formatter(
    "[%(asctime)s] %(levelname)7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
)
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setFormatter(FORMATTER)


def get_logger(name: str) -> logging.Logger:
    """Attach to the default logger."""

    if name in LOGGER_TABLE:
        return LOGGER_TABLE[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(STREAM_HANDLER)

    LOGGER_TABLE[name] = logger
    return logger


def enable_log_file():
    """Add file handler to all loggers."""

    file_handler = logging.FileHandler("run-{}.log".format(get_time_str()))
    file_handler.setFormatter(FORMATTER)

    for logger in LOGGER_TABLE.values():
        logger.addHandler(file_handler)


def disable_stream_handler(func: Callable):
    """Disable stream (console) handler when running a function."""

    def _wrapper(*args, **kwargs):
        for logger in LOGGER_TABLE.values():
            logger.removeHandler(STREAM_HANDLER)
        ret = func(*args, **kwargs)
        for logger in LOGGER_TABLE.values():
            logger.addHandler(STREAM_HANDLER)
        return ret

    return _wrapper
