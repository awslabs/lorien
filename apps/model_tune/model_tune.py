"""
An example of tuning a model with the database.
"""
# pylint: disable=invalid-name

import argparse
import logging
import os
import re
import sys

import numpy as np

import tvm
from lorien.dialect.tvm_dial.autotvm_dial.result import AutoTVMRecords
from lorien.dialect.tvm_dial.autotvm_dial.workload import AutoTVMWorkload
from tvm import autotvm, relay, rpc
from tvm.contrib import graph_executor
from tvm.contrib.utils import tempdir
from tvm.relay.backend import compile_engine


def create_config():
    """Create the config parser of this app."""
    parser = argparse.ArgumentParser(description="Model Deployment")
    parser.add_argument("-t", "--target", required=True, help="The TVM target string")
    parser.add_argument(
        "--remote",
        help="The <IP>:<port> of the remote device for evaluation. "
        "If not presented, the model is evaluated on local CPU/GPU",
    )

    sche_group = parser.add_mutually_exclusive_group()
    sche_group.add_argument("--table", help="Target table name in the DB for querying schedules")
    sche_group.add_argument("--sche", help="The schedule log file")

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--gcv", help="GluonCV model")
    model_group.add_argument("--tf", help="TensorFlow model")
    model_group.add_argument("--pt", help="PyTorch model")
    model_group.add_argument("--test", help="Relay testing model")

    return parser.parse_args()


def query_by_workloads(workloads, table_name):
    ret = []
    for workload in workloads:
        records = AutoTVMRecords(workload.target, workload.get_workload_key())
        records.query(table_name)
        ret.append(records.to_list() if records else [])
    return ret


def query_by_tasks(tasks, table_name):
    workloads = [AutoTVMWorkload.from_task(task) for task in tasks]
    return query_by_workloads(workloads, table_name)


def query_lorien_db(tasks, file_name, table_name):
    """Query the tuning record from Lorien DB."""
    count = 0
    with open(file_name, "w") as filep:
        for task, results in zip(tasks, query_by_tasks(tasks, table_name)):
            if not results:
                print("Missed %s" % str(task))
                continue

            for result in results:
                filep.write("{}\n".format(AutoTVMRecords.encode(result)))
            count += 1
    print("The best schecule config of %d / %d tasks are queried" % (count, len(tasks)))


def get_relay_testing_model(name):
    """Use the testing networks from Relay."""
    from tvm.relay import testing

    dtype = "float32"
    batch_size = 1
    input_shape = (batch_size, 3, 224, 224)

    if "resnet" in name:
        n_layer = int(name.split("-")[1])
        mod, params = testing.resnet.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif "vgg" in name:
        n_layer = int(name.split("-")[1])
        mod, params = testing.vgg.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif name == "mobilenet":
        mod, params = testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "squeezenet_v1.1":
        mod, params = testing.squeezenet.get_workload(
            batch_size=batch_size, version="1.1", dtype=dtype
        )
    elif name == "inception_v3":
        input_shape = (1, 3, 299, 299)
        mod, params = testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, ("data", input_shape)


def get_gcv_model(model_name):
    """Pull a Gluon CV model."""
    import gluoncv as gcv

    model_name = model_name.lower()

    print("Pulling the model from Gluon CV model zoo...")
    shape = (1, 3, 224, 224)
    if model_name.find("inception") != -1:
        shape = (1, 3, 299, 299)
    elif model_name.find("yolo3") != -1:
        shape = (1, 3, 320, 320)
    elif model_name.startswith("ssd"):
        tokens = re.search(r"ssd_(\d+)_", model_name)
        size = int(tokens.group(1))
        shape = (1, 3, size, size)
    net = gcv.model_zoo.get_model(model_name, pretrained=True)
    ret = relay.frontend.from_mxnet(net, shape={"data": shape})
    return ret[0], ret[1], ("data", shape)


def get_tf_model(model_path):
    """Load a TF model from a file: <model_file_path>:<input name>:<image size>"""
    from tvm.relay.frontend.tensorflow_parser import TFParser

    print("Loading TF model from file...")
    path, name, image_shape = model_path.split(":")
    shape = (name, (1, int(image_shape), int(image_shape), 3))
    logging.basicConfig(level=logging.CRITICAL)
    net = TFParser(path).parse()
    ret = relay.frontend.from_tensorflow(net, layout="NCHW", shape={shape[0]: shape[1]})
    logging.basicConfig(level=logging.WARNING)
    return ret[0], ret[1], shape


def get_pt_model(model_name):
    """Pull a PyTorch model from Torch Vision."""
    import torch
    import torchvision

    print("Pull the model from Torch Vision...")
    shape = (1, 3, 224, 224)
    if model_name.find("inception") != -1:
        shape = (1, 3, 299, 299)

    model = getattr(torchvision.models, model_name)(pretrained=True)
    logging.basicConfig(level=logging.CRITICAL)
    model = model.eval()

    trace = torch.jit.trace(model, torch.randn(shape)).float().eval()
    logging.basicConfig(level=logging.WARNING)
    ret = relay.frontend.from_pytorch(trace, [("img", shape)])
    return ret[0], ret[1], ("img", shape)


def main():
    """Main entry function."""
    # Configs
    configs = create_config()
    target = configs.target
    remote_ip = None
    remote_port = None
    if configs.remote is not None:
        assert target.startswith("llvm")
        try:
            remote_ip, remote_port = configs.remote.split(":")
        except ValueError as err:
            print("Failed to parse remote IP and port: %s" % str(err))
            sys.exit(1)

    if configs.gcv:
        mod, params, data_shape = get_gcv_model(configs.gcv)
    elif configs.tf:
        mod, params, data_shape = get_tf_model(configs.tf)
    elif configs.test:
        mod, params, data_shape = get_relay_testing_model(configs.test)
    elif configs.pt:
        mod, params, data_shape = get_pt_model(configs.pt)
    else:
        raise RuntimeError("Unrecognized model format")

    print("Extracting tuning tasks...")
    tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

    # Setup op task schedules.
    sche_file = ""
    if configs.sche and os.path.exists(configs.sche):
        sche_file = configs.sche
        print("Apply the given schedule file %s" % sche_file)
    elif configs.table:
        print("Querying the best results...")
        sche_file = "best_configs.json"
        query_lorien_db(tasks, sche_file, configs.table)
    else:
        sche_file = ""
        print("Use fallback schedule")

    print("Compiling...")
    compile_engine.get().clear()
    ctx = tvm.autotvm.task.DispatchContext.current
    if sche_file:
        tvm.autotvm.task.DispatchContext.current = autotvm.apply_history_best(sche_file)

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build_module.build(mod, target=target, params=params)
    tvm.autotvm.task.DispatchContext.current = ctx

    if remote_ip is not None and remote_port is not None:
        # Export the compiled binary and upload it to the remote device for evaluation.
        tmp = tempdir()
        lib_fname = tmp.relpath("net.tar")
        lib.export_library(lib_fname)

        # Upload the binary and load it remotely.
        remote = rpc.connect(remote_ip, remote_port)
        remote.upload(lib_fname)
        lib = remote.load_module("net.tar")
        dev_ctx = remote.cpu(0)
    else:
        dev_ctx = tvm.cpu(0) if target.startswith("llvm") else tvm.gpu(0)

    print("Evaluating...")
    runtime = graph_executor.create(graph, lib, dev_ctx)
    runtime.set_input(
        data_shape[0], tvm.nd.array(np.random.uniform(size=data_shape[1]).astype("float32"))
    )
    runtime.set_input(**params)

    ftimer = runtime.module.time_evaluator("run", dev_ctx, number=10, repeat=3)
    prof_res = np.array(ftimer().results) * 1000
    print("Median inference time: %.2f ms" % np.median(prof_res))


if __name__ == "__main__":
    main()
