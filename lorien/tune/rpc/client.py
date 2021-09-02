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
RPC client.
"""
import argparse
from typing import List, Optional, Tuple

import rpyc

from ...database.table import check_table
from ...logger import get_logger
from ...util import load_from_yaml
from ..job import JobConfigs

log = get_logger("RPCClient")


class RPCClient:
    """The RPC client."""

    def __init__(self, configs: argparse.Namespace, silent=False):
        """Parse configs to initialize a client.

        Parameters
        ----------
        configs: argparse.Namespace
            The system configuration of RPC tuner.

        silent: bool
            If true, then all messages will be disabled.
        """
        # Parse server string
        server_str = configs.server
        if server_str.find(":") == -1:
            raise RuntimeError("Missing port")

        try:
            port = int(server_str[server_str.find(":") + 1 :])
        except ValueError:
            raise RuntimeError("Invalid port: %s" % server_str[server_str.find(":") + 1 :])
        server_name = server_str[: server_str.find(":")]

        # Connect to the server
        try:
            conn = rpyc.connect(
                server_name,
                port,
                config={
                    "allow_public_attrs": True,
                    "allow_pickle": True,
                    "sync_request_timeout": None,
                },
            )
            if not silent:
                log.info("%s connected", server_str)
        except Exception as err:  # pylint: disable=broad-except
            raise RuntimeError("Failed to connect to %s: %s" % (server_str, str(err)))

        self.target = configs.target
        self.job_configs: Optional[JobConfigs] = None
        self.socket_port = str(conn._channel.stream.sock.getsockname()[1])
        self.conn = conn
        self.token = ""

    def init_worker(self, configs: argparse.Namespace):
        """Initialize the worker with tuning options.

        Parameters
        ----------
        configs: argparse.Namespace
            The system configure for RPC server.
        """

        job_configs_str = self.conn.root.get_job_configs_str(self.token)
        self.job_configs = load_from_yaml(job_configs_str)
        if self.job_configs is None:
            raise RuntimeError(
                "Job configuration has not been initialized on the server "
                "or this work is not registered yet"
            )

        # Check if the AWS credential on this worker can access the DynamoDB table, and remove
        # commit options if failed.
        assert self.job_configs.commit_options is not None
        if not check_table(
            self.job_configs.commit_options["table-name"],
            self.job_configs.commit_options["table-arn"],
            **self.job_configs.commit_options["db"]
        ):
            log.warning("AWS credential is invalid. Will let the master commit results")
            self.job_configs.commit_options = None

        self.job_configs.localize(self.target, configs=configs)

    def init_server(self, job_configs: JobConfigs):
        """Initialize the server options. The client that initializes the server becomes
        the root client, which is authorized to submit jobs and fetch results.

        Parameters
        ----------
        configs: argparse.Namespace
            The system configurations including tune and measure options.

        job_configs: JobConfigs
            The job configurations.
        """
        self.token = self.conn.root.init(self.socket_port, job_configs)

    def submit(self, job_str: str) -> bool:
        """Submit a serialized job to the server. Only the client with root permission
        can submit jobs.

        Parameters
        ----------
        job_str: str
            The serialized job to be submitted.

        Returns
        -------
        success: bool
            True if the job is submitted successfully; False otherwise.
        """
        return self.conn.root.submit(self.token, job_str)

    def fetch_results(self) -> List[Tuple[str, str]]:
        """Fetch tuned results from the server. Only the client with root permission is allowed
        to fetch results.

        Returns
        -------
        results: List[Tuple[str, str]]
            A list of serialized (job, result) pairs.
        """
        return self.conn.root.fetch_results(self.token)

    def is_server_init(self) -> bool:
        """Check if the server is initialized.

        Returns
        -------
        init: bool
            True if all options are ready; False otherwise.
        """
        return self.conn.root.is_init()

    def num_workers(self) -> int:
        """Get the number of live workers.

        Returns
        -------
        n_workers: int
            The number of live workers.
        """
        return self.conn.root.num_workers()

    def register_as_worker(self) -> Tuple[bool, str]:
        """Register client self as a tuning worker.

        Returns
        -------
        token_or_msg: Tuple[bool, str]
            A tuple of (success, token or error message).
        """
        ret = self.conn.root.register_worker(self.socket_port, self.target)
        self.token = ret[1] if ret[0] else None
        return ret

    def request_job(self) -> Optional[str]:
        """Request a job from the server. The result will be stored in cached_result
        so this function will not return anything.
        """
        job_str = self.conn.root.request_job(self.token)
        if not job_str:
            return None

        return job_str

    def send_result(self, job_n_result: Tuple[str, str]) -> str:
        """Send the serailized job and result back to the server.

        Parameters
        ----------
        job_n_result: Tuple[str, str]
            A string pair of job and result.

        Returns
        -------
        msg: str
            The error message.
        """
        return self.conn.root.send_result(self.token, job_n_result)
