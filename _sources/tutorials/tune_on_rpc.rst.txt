..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

.. _tune-on-rpc:

#####################
Tuning on RPC Workers
#####################

In this tutorial, we introduce how to setup Lorien RPC (remote procedure call) servers and use them for tuning. If your target platform is not available on AWS batch (e.g., mobile and edge devices) or you do not want to configure an AWS batch, this is the right place for you.

Lorien RPC tuner is composed of the following components:

- **Tuning Master (RPC server)**: Tuning master could be on any machine that allows public connections. For example, you can launch an EC2 instance with a range of ports opened. Tuning master is in charge of the following tasks:

  1. Load workloads.
  2. Create jobs.
  3. Launch an RPC server.
  4. Accept connections from RPC clients.
  5. Submit jobs to the RPC server.
  6. Track the tuning progress.
  7. Collect results from the RPC server.

- **Workers (RPC client)**: Workers are the machines that actually running the tuning jobs. A brief workflow is shown as follows:

  1. Launch RPC client.
  2. Connect to the tuning master (RPC server).
  3. Register itself as a worker.
  4. Request for a tuning job.
  5. Perform tuning.
  6. Submit results to the database (optional).
  7. Send results back to the tuning master.

For example, if you are tuning on an EC2 instance, then that instance itself would be a worker. If you are tuning on mobile phones, then the host machine connecting to the mobile phones would be a worker. We will explain how to set up workers in each case in the following sections. 

The rest of this tutorial is composed of 3 sections. The first section demonstrates how to configure and launch a tuning master. The second section illustrates how to launch a worker and register it to the master for tuning jobs on an EC2 instance. The third section shows how to launch and register a worker for tuning jobs on a device host with mobile phones.

*************************
Configure A Tuning Master
*************************

Configure a tuning master is relatively straightforward. Here is an example of configure file for a tuning master:

.. code-block:: yaml

  # tune_rpc.yaml
  rpc:
    target: lvm -device=arm_cpu -mtriple=aarch64-linux-gnu
    port: 18871
  tuner: random
  ntrial: 3000
  commit-table-name: lorien-data
  commit-nbest: 3
  commit-log-to: tuned-logs

where ``llvm -device=arm_cpu -mtriple=aarch64-linux-gnu`` indicates the platform we want to tune, and ``port`` will be used to launch an RPC server for workers to connect. It implies that we can also launch multiple tuning masters at the same time by assigning different ports to different targets.

.. note::

    See :ref:`tune-on-aws-batch` for more detail explanations to other tuning paramters.

Then we can use Lorien CLI to launch a tuning master:

.. code-block:: bash

  python3 -m lorien tune @tune_rpc.yaml @workloads.yaml

where ``workloads.yaml`` is a file with workloads we want to tune. Note that if you have no idea about how to prepare the workloads for tuning, you can refer to :ref:`tune-on-local`

After the tuning master is launched, we should see the following messages on console. Since we have not registered any workers, it shows 0 workers for now.

.. code-block:: bash

  INFO Master: Loaded 100 workloads
  INFO Master: Tuning logs on llvm will be uploaded to s3://tuned-logs/topi-llvm
  INFO RPCServer: Launching RPC server at port 18871
  INFO RPCServer: Server has been initialized
  llvm -keys=arm_cpud,cpu -device=arm_cpu -mtriple=aarch64-linux-gnu on 0 RPC workers:   0%|                                    | 0/100 [00:02<?, ?it/s]

At this moment, the tuning master is ready to distribute tuning jobs to workers. In the next section, we are going to launch a worker on a different machine for tuning.

****************************************
Configure A Tuning Worker on An Instance
****************************************

Although most famous EC2 instances are supported by AWS batch so we should be able to use AWS batch tuner to ease our efforts, some new types of instance such as Graviton 2 are not yet supported. In this case, we need to manually launch those instances and register them as tuning workers.

Suppose we have launched a Graviton 2 instance and deployed both TVM and Lorien, what we need to do is running the following command (assuming the tuning master's IP is ``1.2.3.4``):

.. code-block:: bash

    python3 -m lorien rpc-client --server 1.2.3.4:18871 --target "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu"

Then we should see the following responses:

.. code-block:: bash

    INFO RPCClient: Connecting to server 1.2.3.4:18871
    INFO RPCClient: localhost:18871 connected
    INFO RPCClient: Register token ccf7ee15-b891-4dfc-aaa4-f449635063aa
    INFO RPCClient: LocalRunner
    INFO RPCClient: Requesting a job for tuning
    INFO RPCClient: Start tuning
    INFO Tuner: Tuning workload ...skip...

where the register token is the token generated by the RPC server to avoid unauthorized operations. Most communications with the RPC server such as job requesting and result transferring require the token. The token will expire right after the worker is disconnected, which means a worker will need a new token when it reconnects to the server in case it was disconnected accidently. In addition, the message ``LocalRunner`` indicates that this worker will tune jobs locally, meaning that the target platform (ARM CPU with target string ``llvm -device=arm_cpu -mtriple=aarch64-linux-gnu`` in this example) is accessible directly.

Meanwhile, we can also find some changes in the master (RPC server) side:

.. code-block:: bash

  llvm -mcpu=skylake-avx512 on 1 RPC workers:
  0%|                                    | 0/100 [00:31<?, ?it/s]

As a result, we can track the number of activing workers in real time.

******************************************************************
Configure A Tuning Worker on A Machine Connecting to Mobile Phones
******************************************************************

When the target platform is not accessible directly on a host machine (e.g., an ARM CPU with target string ``llvm -mtriple=aarch64-linux-gnu`` used by a mobile phone), the configuration becomes a bit complicate. We need to follow `an AutoTVM tutorial <https://docs.tvm.ai/tutorials/autotvm/tune_relay_mobile_gpu>`_ to set up another connection between the device and the host machine. Specifically, our goal is to set up the following connections:

.. code-block:: bash

    Master: Lorien RPC server
                   |
    Worker: Lorien RPC client, TVM RPC tracker
                                    |
    Device:                    TVM RPC server

As can be seen, the worker (host machine) launches both Lorien RPC client and TVM RPC tracker. When Lorien RPC client successfully requests a tuning job, AutoTVM will be used to tune it. During the tuning, AutoTVM builds an executable binary for the device on the host machine, and sends the binary to the device via TVM RPC tracker for performance measurement. Here are the detail steps for configuring each of them.

1. [Worker] Start A TVM RPC Tracker
-----------------------------------

Use the following command on the host machine to launch a TVM RPC tracker:

.. code-block:: bash

    python3 -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190

2. [Device] Register A Device
-----------------------------

Use the following command on the device to launch a TVM RPC server and register it to the host:

.. code-block:: bash

    python3 -m tvm.exec.rpc_server --tracker=[HOST_IP]:9190 --key=my-device-key

.. note::

    Please refer to `the description <https://docs.tvm.ai/tutorials/autotvm/tune_relay_mobile_gpu.html#register-devices-to-rpc-tracker>`_ in AutoTVM tutorial for deploying a TVM runtime on devices for tuning.

where ``key`` is a label defined by yourself to pair the target. We will use it when configuring the Lorien RPC client in the next step.

3. [Worker] Start A Lorien RPC Client
-------------------------------------

Finally, we start a Lorien RPC client by the following command:

.. code-block:: bash

    python3 -m lorien rpc-client --server 1.2.3.4:18871 --target "lvm -device=arm_cpu -mtriple=aarch64-linux-gnu" --device my-device-key --runner-port 9190

where ``device`` is the device name for pairing; ``runner-port`` is the TVM RPC tracker port we launched in the previous step. As can be seen, this command launches a Lorien RPC client that registers itself to a Lorien tuning master and as well as connects to the TVM RPC tracker.

Again, we would see the following messages:

.. code-block:: bash

    INFO RPCClient: Connecting to server 1.2.3.4:18871
    INFO RPCClient: localhost:18871 connected
    INFO RPCClient: Register token ccf7ee15-b891-4dfc-aaa4-f449635063aa
    INFO RPCClient: RPCRunner: 0.0.0.0:9190 - my-device-key
    INFO RPCClient: Requesting a job for tuning
    INFO RPCClient: Start tuning
    INFO Tuner: Tuning workload ...skip...

*************
Resume Tuning
*************

In case the tuning master was interrupted accidentally, you can relaunch the master with the job trace file to resume the tuning, where the trace file was generated by the job manager automatically. Please note that clients are required to connect to the master again.

.. code-block:: bash

  python3 -m lorien tune @tune_rpc.yaml @workloads.yaml --trace-file=<trace_file_path>

