# Copyright The Lightning AI team.
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

import time
from pathlib import Path

__author__ = "Intel OpenVINO"
__author_email__ = "help@openvino.intel.com"
__license__ = "Apache-2.0"
__copyright__ = f"Copyright (c) 2018-{time.strftime('%Y')}, {__author__}."
__homepage__ = "https://github.com/openvinotoolkit/anomalib"
__docs_url__ = "https://anomalib.readthedocs.io/"
# this has to be simple string, see: https://github.com/pypa/twine/issues/522
__docs__ = "Anomalib  - Anomaly detection library for research and benchmarking."
__long_doc__ = """
Add the long description from the README.md here.
"""

__all__ = [
    "__author__",
    "__author_email__",
    "__copyright__",
    "__docs__",
    "__docs_url__",
    "__homepage__",
    "__license__",
]
