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
The RPC server.
"""
import uuid
from queue import Queue
from typing import Callable, Dict, List, Optional, Tuple

from rpyc import Connection, Service

from ...logger import get_logger
from ...util import dump_to_yaml
from ..job import JobConfigs

log = get_logger("RPCServer")

# Generate a root token for job manager.
ROOT_TOKEN = str(uuid.uuid4())


def need_root(func: Callable):
    """A decorator that ensures RPC functions can only be called by a client with the root token.

    Parameters
    ----------
    func: Callable
        The RPC server methods that requires root permission.

    Returns
    -------
    _check_n_run: Callable
        A function with the an additional argument to accept a client token for checking.
    """

    def _check_n_run(self, token, *args, **kwargs):
        if token == ROOT_TOKEN:
            return func(self, *args, **kwargs)
        raise RuntimeError("No permission to run this function (are you root?)")

    return _check_n_run


class RPCService(Service):
    """The RPC service for tuning AutoTVM tasks."""

    def __init__(self, target: str):
        super(RPCService, self).__init__()

        self.target = target
        self.job_configs: Optional[JobConfigs] = None

        # Map server side connection to the generated token. None if the client has not registered.
        self.conn_to_token: Dict[Connection, Optional[str]] = {}

        self.queue_jobs: Queue = Queue()

        # Map worker token to the job it is running.
        self.worker_n_jobs: Dict[str, Optional[str]] = {}

        # Cached serialized pair of (job, result).
        self.cached_results: List[Tuple[str, str]] = []

    def on_connect(self, conn: Connection):
        """Will be called automatically when a new connection is established.
        We maintain the socket information in order to track the future incoming client requests.
        Especially, when a client is disconnected, we are capable of recycling its job back to
        the job queue.

        .. note::
            We cannot use the socket name as the key to tokens, because the stream will be closed
            prior to call ``on_disconnect`` and the socket name will be unavailable to match the
            token.

        Parameters
        ----------
        conn: Connection
            The new connection on the server side.
        """
        name = str(conn._channel.stream.sock.getpeername())
        log.debug("A new client with socket name %s is connected", name)
        self.conn_to_token[conn] = None

    def on_disconnect(self, conn: Connection):
        """Will be called automatically when a connection object is being cleaned.
        When a connection of a registered worker is cleaned, we invalid its token and recycle
        its running job back to the job queue.

        Parameters
        ----------
        conn: Connection
            The new connection on the server side.
        """
        if not conn in self.conn_to_token:
            log.debug("An unknown client is disconnected")
            return

        token = self.conn_to_token[conn]
        log.debug("A client with token %s is disconnected", token)
        if token is not None and token in self.worker_n_jobs:  # The client is a worker.
            job = self.worker_n_jobs[token]
            if job is not None:
                log.debug("Recycle job: %s", job)
                self.queue_jobs.put(job)
            del self.worker_n_jobs[token]
        del self.conn_to_token[conn]

    def match_socket_port(self, socket_port: str) -> Optional[Connection]:
        """Check if the given socket name matches the list maintained in the server.

        Parameters
        ----------
        socket_port: str
            A string of client socket port. Usually can be retrieved by
            `client_conn._channel.stream.sock.getsockname()[1]`.

        Returns
        -------
        conn: Optional[Connection]
            The corresponding server side connection. None if no match.
        """
        for conn in self.conn_to_token:
            port = str(conn._channel.stream.sock.getpeername()[1])
            if socket_port == port:
                return conn
        return None

    def init(self, socket_port: str, job_configs: JobConfigs) -> str:
        """Initialize job configurations. The client that calls this method will acquire
        the root token to call other methods that requires root permission.

        Parameters
        ----------
        socket_port: str
            A string of client socket port. Usually can be retrieved by
            `client_conn._channel.stream.sock.getsockname()[1]`.

        job_configs: JobConfigs
            The job configurations.
        """
        server_conn = self.match_socket_port(socket_port)
        if server_conn is None:
            raise RuntimeError(
                "Socket port %s does not match any connections in the server" % socket_port
            )

        assert not self.is_init(), "The server has already been initialized"

        self.job_configs = job_configs

        log.info("Server has been initialized")
        self.conn_to_token[server_conn] = ROOT_TOKEN
        return ROOT_TOKEN

    def is_init(self) -> bool:
        """Check if the server is initialized.

        Returns
        -------
        init: bool
            True if job configuration is ready; False otherwise.
        """
        return self.job_configs is not None

    @need_root
    def submit(self, job_str: str) -> bool:
        """Accept a job for tuning. Since the job queue in RPC server has no backup,
        we want to keep it minimized. As a result, we only accept at most worker number of jobs
        in the queue to be the next job for each worker.

        Parameters
        ----------
        job: str
            A job string.

        Returns
        -------
        success: bool
            True if the job is submitted successfully; False otherwise.
        """
        if self.queue_jobs.qsize() < len(self.worker_n_jobs):
            self.queue_jobs.put(job_str)
            return True
        return False

    @need_root
    def fetch_results(self) -> List[Tuple[str, str]]:
        """Fetch and clean the tuned results.

        Returns
        -------
        results: List[Tuple[str, str]]
            A list of serialized (job, result) pairs.
        """
        ret = self.cached_results
        self.cached_results = []
        return ret

    def num_workers(self) -> int:
        """Get the number of live workers.

        Returns
        -------
        n_workers: int
            The number of live workers.
        """
        return len(self.worker_n_jobs)

    def register_worker(self, socket_port: str, target: str) -> Tuple[bool, str]:
        """Let client register itself as a tuning worker.

        Parameters
        ----------
        socket_port: str
            A string of client socket port. Usually can be retrieved by
            `client_conn._channel.stream.sock.getsockname()[1]`.

        target: str
            The client expected the target string for checking.

        Returns
        -------
        token_or_msg: Tuple[bool, str]
            A tuple of (success, token or error message).
        """
        server_conn = self.match_socket_port(socket_port)
        assert (
            server_conn is not None
        ), "Socket port {} does not match any connections in the server".format(socket_port)

        if target != self.target:
            return (
                False,
                "Server is expecting clients to tune {0}, but got {1}".format(self.target, target),
            )

        token = str(uuid.uuid4())
        self.conn_to_token[server_conn] = token
        self.worker_n_jobs[token] = None
        log.debug("New worker registered: %s", token)
        return (True, token)

    def get_job_configs_str(self, token: str) -> str:
        """Request job configs in string (YAML format).

        Returns
        -------
        token: str
            The client token.

        job_configs_str: str
            The job configs in string.
        """
        if not token in self.worker_n_jobs:
            log.error("%s is not a registered worker token!", token)
            return ""

        if self.job_configs is None:
            log.error("Job configuration has not been initialized")
        return dump_to_yaml(self.job_configs)

    def request_job(self, token: str) -> str:
        """Request one job to tune.

        Parameters
        ----------
        token: str
            The client token.

        Returns
        -------
        job: str
            The job in string format.
        """
        if not token in self.worker_n_jobs:
            log.error("%s is not a registered worker token!", token)
            return ""

        if self.queue_jobs.empty():
            return ""

        job = self.queue_jobs.get()
        self.worker_n_jobs[token] = job
        return job

    def send_result(self, token: str, job_n_result: Tuple[str, str]) -> str:
        """Accept a serialized result from workers.

        Parameters
        ----------
        token: str
            The client token.

        job_n_result: Tuple[str, str]
            A string pair of job and result.

        Returns
        -------
        msg: str
            The result message for client.
        """
        if not token in self.worker_n_jobs:
            return "{} is not a registered worker token".format(token)

        if job_n_result[0] != self.worker_n_jobs[token]:
            return "{} is not allocated to tune the job".format(token)

        self.cached_results.append(job_n_result)
        self.worker_n_jobs[token] = None
        return ""
