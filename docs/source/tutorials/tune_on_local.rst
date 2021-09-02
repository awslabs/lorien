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

.. _tune-on-local:

###############
Tuning on Local
###############

In this tutorial, we introduce how to use Lorien to tune tasks locally step-by-step. This could be a quick way to let you hand on with Lorien to see what it can do. Specifically, we demonstrate how to extract tuning tasks from ResNet-18 in GluonCV model zoo and tune them using AutoTVM on the local machine.

*****
Steps
*****

1. Setup Lorien
---------------

Please first refer to :ref:`setup` to setup Lorien.

2. Extract tuning tasks
-----------------------

Lorien can be configured via commandline or a configure file in YAML format. Here is an example configure file of extracting workloads from ResNet-18 in GluonCV model zoo (note that MXNet and GluonCV must be available in your environment):

.. code-block:: yaml

    gcv:
        - resnet18_v1
    output:
        - workloads.yaml

where ``gcv`` indicates that the model file is coming from GluonCV model zoo. You can find some samples in configs/samples. In addition, Lorien also supports model files in other framework formats. You can run the following command to check the supported formats:

.. code-block:: bash

    python3 -m lorien generate autotvm extract-from-model -h

Then we simply run the command with the configure file to extract workloads:

.. code-block:: bash

    python3 -m lorien generate autotvm extract-from-model @configs/gcv_resnet18.yaml --target llvm

The above command outputs a file ``workloads.yaml``. Each line in this file indicates a tunable workload. In the case of AutoTVM, it looks like the following

.. code-block:: yaml

    - '!!python/object:lorien.dialect.tvm_dial.autotvm_dial.workload.AutoTVMWorkload {_primary_key: null, args: [[TENSOR, [1, 256, 14, 14], float32], [TENSOR, [512, 256, 1, 1], float32], [2, 2], [0, 0, 0, 0], [1, 1], NCHW, NCHW, float32], target: llvm -keys=cpu -link-params=0, task_name: conv2d_NCHWc.x86}'

This is a serialized YAML object of an AutoTVM workload. If you are using a different dialect, you may see a completely different format in the workload file.


3. Tune the Workloads
---------------------

To tune the workloads we just extracted, we again need to have a configure file as follows:

.. code-block:: yaml

    # Tuning options.
    local: llvm
    db:
        - endpoint_url: http://localhost:10020
    tuner: random
    ntrial: 15

    # We enable clflush for x86 targets so we can have fewer tests.
    test: 1
    repeat: 10
    min: 1

    # Result committing options.
    commit-nbest: 1
    commit-table-name: lorien
    # Uncomment this line if you have configured AWS CLI and S3 bucket.
    #commit-log-to: saved-tuning-logs

    where ``ntrial`` indicates how many tuning trials for each workload; ``test``, ``repeat``, ``min`` are the configurations for schedule candidate evaluation. In the above configures, we run each schedule candidate once to get its execution latency, and repeat this process 10 times to eliminate the variants. Note that ``min`` means "minimum repeat time in ms". It means the total run time of 10 runs is less than 1 ms, then we keep repeating until the total run time is 1 ms. This is also used to eliminate the variants.

    In addition, in result committing options, ``commit-nbest`` means we will commit the best 1 schedule of each task to the table ``lorien`` in DynamoDB. Also, if you have configured AWS credential so that Lorien can access your S3 buckets via AWS CLI, you can let Lorien upload the complete tuning logs (with 15 explored schedules in this example) to the S3 bucket. These logs can be used to train a performance cost model later.
    
    Finally, ``db`` configres the DynamoDB. Again, if you have configured AWS credential so that Lorien can access DynamoDB via AWS CLI, you can get rid of the ``db`` endpoint configuration. In this tutorial, we will launch a local DynamoDB for demonstraction purpose (Jave Runtime Environment is required). Specifically, you could open another terminal and run the following command:

.. code-block:: bash

    make launch_local_db

This command launches a local DynamoDB at the port 10020. It is now ready to receive queries via endpoint http://localhost:10020.

Now we can start tuning:

.. code-block:: bash

  python3 -m lorien tune @tune_local.yaml @gcv_workloads.yaml

Since we tune workloads locally, we will directly find the complete tuning logs in the current directly. You will see a directory with ``lorien-tune-log-`` prefix. Each file in the directory is the tuning log of a task.

4. Check Results
----------------

Now we use Lorien APIs to check if the best schedules has been correctly committed to the local DynamoDB:

.. code-block:: python

    >>> from lorien import database
    >>> database.list_tables(endpoint_url="http://localhost:10020")
    ['lorien']
    >>> data = list(database.table.scan_table("lorien", endpoint_url="http://localhost:10020"))
    >>> len(data[0]["Items"])
    # The number of tuned tasks.
    >>> data[0]["Items"][0]["TargetIDKeys"]
    {'S': 'llvm_cpu'}
    >>> data[0]["Items"][0]["PrimaryRangeKey"]
    {'S': 'conv2d_NCHWc.x86#_TENSOR__1_256_14_14__float32_#_TENSOR__256_256_3_3__float32_#_1_1_#_1_1_1_1_#_1_1_#NCHW#NCHW#float32'}
    >>> len(data[0]["Items"][0]["BestConfigs"])
    1

Success! It means the tuned schedules have been maintained in the DynamoDB. As a result, we can use the query API to query them when building the model. For simplify, we directly use the workload key we got above to query the schedule. In practice, you should extract workload from the model you are building, and use the workload key to query the best schedule.

.. code-block:: python

    >>> from lorien.dialect.tvm_dial.autotvm_dial.result import AutoTVMRecords
    >>> records = AutoTVMRecords("llvm", "conv2d_NCHWc.x86#_TENSOR__1_256_14_14__float32_#_TENSOR__256_256_3_3__float32_#_1_1_#_1_1_1_1_#_1_1_#NCHW#NCHW#float32")
    >>> records.query("lorien", endpoint_url="http://localhost:10020")
    >>> len(records)
    5
    >>> records.peak()
    (MeasureInput(target=llvm -keys=cpu -link-params=0, task=Task(func_name=conv2d_NCHWc.x86, args=(('TENSOR', (1, 256, 14, 14), 'float32'), ('TENSOR', (256, 256, 3, 3), 'float32'), (1, 1), (1, 1, 1, 1), (1, 1), 'NCHW', 'NCHW', 'float32'), kwargs={}, workload=('conv2d_NCHWc.x86', ('TENSOR', (1, 256, 14, 14), 'float32'), ('TENSOR', (256, 256, 3, 3), 'float32'), (1, 1), (1, 1, 1, 1), (1, 1), 'NCHW', 'NCHW', 'float32')), config=[('tile_ic', [-1, 256]), ('tile_oc', [-1, 1]), ('tile_ow', [-1, 4]), ('unroll_kw', True)],None,170), MeasureResult(costs=(0.005246509, 0.005248521, 0.0052774960000000004, 0.005288638, 0.0053149600000000005, 0.005316861, 0.005321088999999999, 0.00532152), error_no=0, all_cost=1.108623743057251, timestamp=1624993541.8088477))

5. Fault Tolerance
------------------

All state changes of tuing jobs will be recorded in a ``lorien-tune-<timestampe>.trace`` file. In case the master was interrupted and you wish to resume the tuning, you could specify ``--trace-file`` in the command, so that the tuning master will skip the finished jobs and keep tracking the state of tuning jobs.

.. code-block:: bash
   
  python3 -m lorien tune @tune_local.yaml @gcv_workloads.yaml --trace-file=<trace_file_path>
