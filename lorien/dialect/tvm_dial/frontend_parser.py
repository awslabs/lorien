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
The module of TVM workload extraction from models.
"""
import argparse
import logging
from typing import Any, Callable, Dict, List, Tuple

import tvm
from tvm import relay

from ...logger import get_logger
from ...util import load_from_yaml

log = get_logger("Extractor")
EXTRACTOR_FUNC_TABLE: Dict[str, Callable] = {}


def register_extractor(framework: str) -> Callable:
    """Register extractor for a target ML framework.

    Parameters
    ----------
    framework: str
        Framework's registration name.

    Returns
    -------
    reg: Callable
        A callable function for registration.
    """

    def _do_reg(func: Callable):
        if framework in EXTRACTOR_FUNC_TABLE:
            raise RuntimeError("Config parser of %s has been registered" % framework)

        EXTRACTOR_FUNC_TABLE[framework] = func

    return _do_reg


def parse_model_config(
    model_desc: str, default_shape: Dict[str, Tuple[int, ...]]
) -> Tuple[str, Dict[str, Tuple[int, ...]]]:
    """A helper function to parse model file path and input shape.

    Parameters
    ----------
    model_desc: str
        Model description in YAML format.

    default_shape: Tuple[int, ...]
        The default shape for this model if shape is not specified in `model_desc`.

    Returns
    -------
    model_path_n_shape: Tuple[str, Dict[str, Tuple[int, ...]]]
        Model file path or name in the online model zoo.
    """
    model = load_from_yaml(model_desc)
    if isinstance(model, str):  # No shape specified. Use default shape.
        model_path = model
        shape = default_shape
    elif isinstance(model, dict) and len(model) == 1 and isinstance(list(model.values())[0], dict):
        model_path = list(model.keys())[0]
        shape = model[model_path]
    else:
        raise RuntimeError("Unrecognized model description: %s" % model)
    return model_path, shape


@register_extractor("gcv")
def parse_from_gcv(
    configs: argparse.Namespace,
) -> List[Tuple[str, Tuple[tvm.IRModule, Dict[str, Any]]]]:
    """Extract workloads with Gluon CV model zoo.

    Parameters
    ----------
    configs: argparse.Namespace
        The system configure of generate.extract.

    Returns
    -------
    mod_n_params: List[Tuple[str, Tuple[tvm.IRModule, Dict[str, Any]]]]
        A list of (model name, (Relay module, parameters)).
    """
    import gluoncv

    # Process models.
    mod_n_params: List[Tuple[str, Tuple[tvm.IRModule, Dict[str, Any]]]] = []
    failed: List[Tuple[str, str]] = []

    log.info("Parsing %d Gluon CV models", len(configs.gcv))
    for model_desc in configs.gcv:
        model_name, shape = parse_model_config(model_desc, {"data": (1, 3, 224, 224)})
        try:
            net = gluoncv.model_zoo.get_model(model_name, pretrained=True)
            mod_n_params.append((model_name, relay.frontend.from_mxnet(net, shape=shape)))
        except Exception as err:  # pylint:disable=broad-except
            failed.append((model_name, str(err)))
            continue

    for model, msg in failed:
        log.warning("Failed to load and convert the Gluon CV model %s: %s", model, msg)

    log.info("Collecting workloads from %d Gluon CV models", len(mod_n_params))
    return mod_n_params


@register_extractor("mxnet")
def parse_from_mxnet(
    configs: argparse.Namespace,
) -> List[Tuple[str, Tuple[tvm.IRModule, Dict[str, Any]]]]:
    """Extract workloads from MXNet models.

    Parameters
    ----------
    configs: argparse.Namespace
        The system configure of generate.extract.

    Returns
    -------
    mod_n_params: List[Tuple[str, Tuple[tvm.IRModule, Dict[str, Any]]]]
        A list of (model name, (Relay module, parameters)).
    """
    import mxnet

    # Process models.
    mod_n_params: List[Tuple[str, Tuple[tvm.IRModule, Dict[str, Any]]]] = []
    failed: List[Tuple[str, str]] = []

    log.info("Parsing %d MXNet models", len(configs.mxnet))
    for model_desc in configs.mxnet:
        model_path, shape = parse_model_config(model_desc, {"data": (1, 3, 224, 224)})

        try:
            symbol = mxnet.sym.load("{}-symbol.json".format(model_path))
            mx_inputs = [mxnet.sym.var(name) for name in shape.keys()]
            net = mxnet.gluon.SymbolBlock(symbol, mx_inputs)
            net.hybridize()
            net.collect_params().load("{}-0000.params".format(model_path))
            mod_n_params.append((model_path, relay.frontend.from_mxnet(net, shape=shape)))
        except Exception as err:  # pylint:disable=broad-except
            failed.append((model_path, str(err)))
            continue

    for model, msg in failed:
        log.warning("Failed to load and convert the MXNet model %s: %s", model, msg)

    log.info("Collecting workloads from %d MXNet models", len(mod_n_params))
    return mod_n_params


@register_extractor("keras")
def parse_from_keras(
    configs: argparse.Namespace,
) -> List[Tuple[str, Tuple[tvm.IRModule, Dict[str, Any]]]]:
    """Extract workloads from Keras models.

    Parameters
    ----------
    configs: argparse.Namespace
        The system configure of generate.extract.

    Returns
    -------
    mod_n_params: List[Tuple[str, Tuple[tvm.IRModule, Dict[str, Any]]]]
        A list of (model name, (Relay module, parameters)).
    """
    from keras.models import load_model

    mod_n_params: List[Tuple[str, Tuple[tvm.IRModule, Dict[str, Any]]]] = []
    failed: List[Tuple[str, str]] = []

    log.info("Parsing %d Keras models", len(configs.keras))
    for model_desc in configs.keras:
        model_path, shape = parse_model_config(model_desc, {"input_1": (1, 3, 224, 224)})
        try:
            keras_model = load_model(model_path)
            logging.disable(logging.CRITICAL)
            mod_n_params.append((model_path, relay.frontend.from_keras(keras_model, shape=shape)))
            logging.disable(logging.NOTSET)
        except Exception as err:  # pylint:disable=broad-except
            failed.append((model_path, str(err)))
            continue

    for model, msg in failed:
        log.warning("Failed to load and convert the Keras model %s: %s", model, msg)
    log.info("Collecting workloads from %d Keras models", len(mod_n_params))
    return mod_n_params


@register_extractor("onnx")
def parse_from_onnx(
    configs: argparse.Namespace,
) -> List[Tuple[str, Tuple[tvm.IRModule, Dict[str, Any]]]]:
    """Extract workloads from ONNX models.

    Parameters
    ----------
    configs: argparse.Namespace
        The system configure of generate.extract.

    Returns
    -------
    mod_n_params: List[Tuple[str, Tuple[tvm.IRModule, Dict[str, Any]]]]
        A list of (model name, (Relay module, parameters)).
    """
    import onnx

    # Process models.
    mod_n_params: List[Tuple[str, Tuple[tvm.IRModule, Dict[str, Any]]]] = []
    failed: List[Tuple[str, str]] = []

    log.info("Parsing %d ONNX models", len(configs.onnx))
    for model_desc in configs.onnx:
        model_path, shape = parse_model_config(model_desc, {"input": (1, 3, 224, 224)})
        try:
            onnx_model = onnx.load(model_path)
            logging.disable(logging.CRITICAL)
            mod_n_params.append((model_path, relay.frontend.from_onnx(onnx_model, shape=shape)))
            logging.disable(logging.NOTSET)
        except Exception as err:  # pylint:disable=broad-except
            failed.append((model_path, str(err)))
            continue

    for model, msg in failed:
        log.warning("Failed to load and convert the ONNX model %s: %s", model, msg)

    log.info("Collecting workloads from %d ONNX models", len(mod_n_params))
    return mod_n_params


@register_extractor("torch")
def parse_from_pytorch(
    configs: argparse.Namespace,
) -> List[Tuple[str, Tuple[tvm.IRModule, Dict[str, Any]]]]:
    """Extract workloads with PyTorch model zoo.

    Parameters
    ----------
    configs: argparse.Namespace
        The system configure of generate.extract.

    Returns
    -------
    mod_n_params: List[Tuple[str, Tuple[tvm.IRModule, Dict[str, Any]]]]
        A list of (model name, (Relay module, parameters)).
    """
    import torch
    import torchvision

    # Process models.
    mod_n_params: List[Tuple[str, Tuple[tvm.IRModule, Dict[str, Any]]]] = []
    failed: List[Tuple[str, str]] = []

    log.info("Parsing %d PyTorch models", len(configs.torch))
    for model_desc in configs.torch:
        model_name, shape_dict = parse_model_config(model_desc, {"img": (1, 3, 224, 224)})
        try:
            model = getattr(torchvision.models, model_name)(pretrained=True)
            logging.disable(logging.CRITICAL)
            model = model.eval()
            if len(shape_dict) > 1:
                raise Exception("Do not support PyTorch model with multiple inputs.")

            shapes = list(shape_dict.values())
            # Ignore type check due to mypy issue https://github.com/python/mypy/issues/8210
            trace = (
                torch.jit.trace(model, [torch.randn(shape) for shape in shapes])  # type: ignore
                .float()
                .eval()
            )

            mod_n_params.append(
                (model_name, relay.frontend.from_pytorch(trace, list(shape_dict.items())))
            )
            logging.disable(logging.NOTSET)
        except Exception as err:  # pylint:disable=broad-except
            failed.append((model_name, str(err)))
            continue

    for model, msg in failed:
        log.warning("Failed to load and convert the PyTorch model %s: %s", model, msg)
    log.info("Collecting workloads from %d PyTorch models", len(mod_n_params))
    return mod_n_params


@register_extractor("tf")
def parse_from_tf(
    configs: argparse.Namespace,
) -> List[Tuple[str, Tuple[tvm.IRModule, Dict[str, Any]]]]:
    """Extract workloads from TensorFlow models.

    Parameters
    ----------
    configs: argparse.Namespace
        The system configure of generate.extract.

    Returns
    -------
    mod_n_params: List[Tuple[str, Tuple[tvm.IRModule, Dict[str, Any]]]]
        A list of (model name, (Relay module, parameters)).
    """
    from tvm.relay.frontend.tensorflow_parser import TFParser

    # Process models.
    mod_n_params: List[Tuple[str, Tuple[tvm.IRModule, Dict[str, Any]]]] = []
    failed: List[Tuple[str, str]] = []

    log.info("Parsing %d TensorFlow models", len(configs.tf))
    for model_desc in configs.tf:
        model_path, shape = parse_model_config(model_desc, {"Placeholder": (1, 224, 224, 3)})
        try:
            net = TFParser(model_path).parse()
            logging.disable(logging.CRITICAL)
            mod_n_params.append(
                (model_path, relay.frontend.from_tensorflow(net, layout="NCHW", shape=shape))
            )
            logging.disable(logging.NOTSET)
        except Exception as err:  # pylint:disable=broad-except
            failed.append((model_path, str(err)))
            continue

    for model, msg in failed:
        log.warning("Failed to load and convert the TF model %s: %s", model, msg)
    log.info("Collecting workloads from %d TensorFlow models", len(mod_n_params))
    return mod_n_params


@register_extractor("tflite")
def parse_from_tflite(
    configs: argparse.Namespace,
) -> List[Tuple[str, Tuple[tvm.IRModule, Dict[str, Any]]]]:
    """Extract workloads from TFLite models.

    Parameters
    ----------
    configs: argparse.Namespace
        The system configure of generate.extract.

    Returns
    -------
    mod_n_params: List[Tuple[str, Tuple[tvm.IRModule, Dict[str, Any]]]]
        A list of (model name, (Relay module, parameters)).
    """
    from tflite.Model import Model

    # Process models.
    mod_n_params: List[Tuple[str, Tuple[tvm.IRModule, Dict[str, Any]]]] = []
    failed: List[Tuple[str, str]] = []

    log.info("Parsing %d TFLite models", len(configs.tflite))
    for model_desc in configs.tflite:
        model_path, shape_dict = parse_model_config(model_desc, {"Placeholder": (1, 224, 224, 3)})
        try:
            with open(model_path, "rb") as f:
                tflite_model_buf = f.read()
            tflite_model = Model.GetRootAsModel(tflite_model_buf, 0)
            dtype_dict = {}
            for input_name in shape_dict:
                dtype_dict[input_name] = "float32"
            mod, params = relay.frontend.from_tflite(
                tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict
            )
            mod = relay.transform.RemoveUnusedFunctions()(mod)
            mod = relay.transform.InferType()(mod)
            mod = relay.transform.ConvertLayout("NCHW")(mod)
            mod_n_params.append((model_path, (mod, params)))
        except Exception as err:  # pylint:disable=broad-except
            failed.append((model_path, str(err)))
            continue

    for model, msg in failed:
        log.warning("Failed to load and convert the TFLite model %s: %s", model, msg)

    log.info("Collecting workloads from %d TFLite models", len(mod_n_params))
    return mod_n_params
