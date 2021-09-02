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
The module to launch RPC client or server.
"""
import argparse
import threading
import signal
import time
from typing import Optional

from rpyc.utils.server import ThreadedServer

from ...configs import create_config_parser, register_config_parser
from ...logger import get_logger
from ...util import dump_to_yaml, load_from_yaml
from ..job import Job
from ..result import TuneErrorCode, TuneResult
from .client import RPCClient
from .server import RPCService

log = get_logger("RPC")


def launch_server(port: int, target: str) -> None:
    """Launch a RPC server.

    Parameters
    ----------
    port: int
        The port for launching the server.

    target: str
        The target string for this server.
    """
    s = ThreadedServer(
        RPCService(target),
        port=port,
        protocol_config={"allow_public_attrs": True, "allow_pickle": True},
    )
    log.info("Launching RPC server at port %d", port)

    try:
        s.start()
    except Exception as err:  # pylint: disable=broad-except
        log.info("RPC server at port %d throws exceptions: %s", port, str(err))

    log.info("RPC server at port %d is shutdown", port)


class ClientThread(threading.Thread):
    """The custom thread to run the client loop."""

    def __init__(self, configs):
        super(ClientThread, self).__init__(name="ClientThread", daemon=True)
        self.configs = configs
        self._stop_event = threading.Event()
        self._sleep_period = 1.0

    def run(self):
        log.info("Connecting to server %s", self.configs.server)
        while True:
            try:
                client = RPCClient(self.configs)
                break
            except Exception as err:  # pylint: disable=broad-except
                log.warning("Failed to connect: %s. Reconnectiong", str(err))
                time.sleep(1)

        success, msg = client.register_as_worker()
        if not success:
            raise RuntimeError("Failed to register as a worker: %s" % msg)
        log.info("Register token %s", client.token)

        client.init_worker(self.configs)
        assert client.job_configs is not None

        while not self._stop_event.isSet():
            log.info("Requesting a job for tuning")
            try:
                job_str = client.request_job()
            except EOFError:
                log.info("Lost server connection")
                break

            if not job_str:
                log.info("Server job queue empty")
                time.sleep(1)
                continue

            log.info("Start tuning")
            job: Optional[Job] = None
            result = TuneResult()
            try:
                job = load_from_yaml(job_str)
            except RuntimeError as err:
                msg = "Failed to create a job {0} from string: {1}".format(job_str, str(err))
                log.warning(msg)
                result.error_code = TuneErrorCode.FAIL_TO_LOAD_WORKLOAD
                result.error_msgs.append(msg)

            if job is not None:
                job.tune(
                    client.job_configs.tune_options,
                    client.job_configs.measure_options,
                    client.job_configs.commit_options,
                )
                result = job.result

            # Send the result back to the server.
            log.info("Result: %s", str(result))
            try:
                msg = client.send_result((job_str, dump_to_yaml(result)))
                if msg:
                    log.error(msg)
            except EOFError:
                log.info("Lost server connection")
                break

            self._stop_event.wait(self._sleep_period)

    def join(self, timeout=None):
        self._stop_event.set()
        threading.Thread.join(self, timeout)


def launch_client(configs: argparse.Namespace) -> None:
    """Launch a RPC client.

    Parameters
    ----------
    configs: argparse.Namespace
        The system configure for RPC server.
    """

    client_thread = ClientThread(configs)
    client_thread.start()

    def signal_handler(sig, fname):  # pylint: disable=unused-argument
        log.info("Ctrl+C pressed")
        client_thread.join()

    signal.signal(signal.SIGINT, signal_handler)
    running = threading.Event()
    running.wait()


@register_config_parser("top.rpc-client")
def define_config() -> argparse.ArgumentParser:
    """Define the command line interface for RPC client.

    Returns
    -------
    parser: argparse.ArgumentParser
        The defined argument parser.
    """
    parser = create_config_parser("Launch RPC client on this machine and connect to the server")
    parser.add_argument(
        "--server", type=str, required=True, help="RPC Server IP and port (e.g., 0.0.0.0:18871)"
    )
    parser.add_argument(
        "--target", type=str, required=True, help="The target string for this client"
    )
    parser.set_defaults(entry=launch_client)
    return parser
