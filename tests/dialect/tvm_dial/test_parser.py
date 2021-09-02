"""
The unit test module for TVM dialect frontend parsers.
"""
# pylint:disable=missing-docstring, redefined-outer-name, invalid-name
# pylint:disable=unused-argument, unused-import, wrong-import-position, ungrouped-imports
import argparse
import mock
import pytest

from lorien.util import is_dialect_enabled

if not is_dialect_enabled("tvm"):
    pytest.skip("TVM dialect is not available", allow_module_level=True)

from lorien.dialect.tvm_dial.frontend_parser import EXTRACTOR_FUNC_TABLE


def test_parse_from_gcv(mocker):
    # Mock GCV frontend and assume it's always working.
    mocker.patch("gluoncv.model_zoo.get_model").return_value = "FakeNet"
    mocker.patch(
        "lorien.dialect.tvm_dial.frontend_parser.relay.frontend.from_mxnet"
    ).return_value = ("mod", "params")

    configs = argparse.Namespace(gcv=["alexnet"], target=["llvm"])

    mod_n_params = EXTRACTOR_FUNC_TABLE["gcv"](configs)
    assert len(mod_n_params) == 1
    assert mod_n_params[0][0] == "alexnet"
    assert mod_n_params[0][1][0] == "mod"
    assert mod_n_params[0][1][1] == "params"

    mocker.patch("gluoncv.model_zoo.get_model").side_effect = Exception("Mocked Error")
    mod_n_params = EXTRACTOR_FUNC_TABLE["gcv"](configs)
    assert len(mod_n_params) == 0


def test_parse_from_keras(mocker):
    # Mock Keras frontend and assume it's always working.
    mocker.patch("keras.models.load_model").return_value = "FakeNet"
    mocker.patch(
        "lorien.dialect.tvm_dial.frontend_parser.relay.frontend.from_keras"
    ).return_value = ("mod", "params")

    configs = argparse.Namespace(keras=["alexnet: { data: [1, 3, 224, 224]}"], target="llvm")

    mod_n_params = EXTRACTOR_FUNC_TABLE["keras"](configs)
    assert len(mod_n_params) == 1
    assert mod_n_params[0][0] == "alexnet"
    assert mod_n_params[0][1][0] == "mod"
    assert mod_n_params[0][1][1] == "params"

    mocker.patch("keras.models.load_model").side_effect = Exception("Mocked Error")
    mod_n_params = EXTRACTOR_FUNC_TABLE["keras"](configs)
    assert len(mod_n_params) == 0


def test_parse_from_onnx(mocker):
    # Mock ONNX frontend and assume it's always working.
    mocker.patch("onnx.load").return_value = "FakeNet"
    mocker.patch(
        "lorien.dialect.tvm_dial.frontend_parser.relay.frontend.from_onnx"
    ).return_value = ("mod", "params")

    configs = argparse.Namespace(onnx=["alexnet: { data: [1, 3, 224, 224]}"], target=["llvm"])

    mod_n_params = EXTRACTOR_FUNC_TABLE["onnx"](configs)
    assert len(mod_n_params) == 1
    assert mod_n_params[0][0] == "alexnet"
    assert mod_n_params[0][1][0] == "mod"
    assert mod_n_params[0][1][1] == "params"

    mocker.patch("onnx.load").side_effect = Exception("Mocked Error")
    mod_n_params = EXTRACTOR_FUNC_TABLE["onnx"](configs)
    assert len(mod_n_params) == 0


def test_extract_from_torch():
    configs = argparse.Namespace(
        torch=["alexnet: { data: [1, 3, 224, 224]}"],
        target=["llvm"],
    )
    mod_n_params = EXTRACTOR_FUNC_TABLE["torch"](configs)
    assert len(mod_n_params) == 1

    configs = argparse.Namespace(
        torch=["alexnet_wrong_name"],
        target=["llvm"],
    )
    mod_n_params = EXTRACTOR_FUNC_TABLE["torch"](configs)
    assert len(mod_n_params) == 0


def test_parse_from_tf(mocker):
    # Mock TensorFlow frontend and assume it's always working.
    mocker.patch(
        "tvm.relay.frontend.tensorflow_parser.TFParser"
    ).return_value.parse.return_value = "FakeNet"
    mocker.patch(
        "lorien.dialect.tvm_dial.frontend_parser.relay.frontend.from_tensorflow"
    ).return_value = ("mod", "params")

    configs = argparse.Namespace(tf=["alexnet: { data: [1, 224, 224, 3]}"], target=["llvm"])

    mod_n_params = EXTRACTOR_FUNC_TABLE["tf"](configs)
    assert len(mod_n_params) == 1
    assert mod_n_params[0][0] == "alexnet"
    assert mod_n_params[0][1][0] == "mod"
    assert mod_n_params[0][1][1] == "params"

    mocker.patch("tvm.relay.frontend.tensorflow_parser.TFParser").side_effect = Exception(
        "Mocked Error"
    )
    mod_n_params = EXTRACTOR_FUNC_TABLE["tf"](configs)
    assert len(mod_n_params) == 0


def test_parse_from_tflite(mocker):
    # Mock tflite frontend and assume it's always working.
    mocker.patch("lorien.dialect.tvm_dial.frontend_parser.open")
    mocker.patch("tflite.Model.Model.GetRootAsModel").return_value.parse.return_value = "FakeNet"
    mocker.patch(
        "lorien.dialect.tvm_dial.frontend_parser.relay.frontend.from_tflite"
    ).return_value = ("mod", "params")

    def dummy_func(a):
        return a

    mocker.patch(
        "lorien.dialect.tvm_dial.frontend_parser.relay.transform.RemoveUnusedFunctions"
    ).return_value = dummy_func
    mocker.patch(
        "lorien.dialect.tvm_dial.frontend_parser.relay.transform.InferType"
    ).return_value = dummy_func
    mocker.patch(
        "lorien.dialect.tvm_dial.frontend_parser.relay.transform.ConvertLayout"
    ).return_value = dummy_func

    configs = argparse.Namespace(tflite=["alexnet: { data: [1, 224, 224, 3]}"], target=["llvm"])

    mod_n_params = EXTRACTOR_FUNC_TABLE["tflite"](configs)
    assert len(mod_n_params) == 1
    assert mod_n_params[0][0] == "alexnet"
    assert mod_n_params[0][1][0] == "mod"
    assert mod_n_params[0][1][1] == "params"

    mocker.patch("tflite.Model.Model.GetRootAsModel").side_effect = Exception("Mocked Error")
    mod_n_params = EXTRACTOR_FUNC_TABLE["tflite"](configs)
    assert len(mod_n_params) == 0


def test_parse_from_mxnet(mocker):
    mocker.patch("mxnet.sym.load").return_value = None

    dummy_sym = mock.MagicMock()
    dummy_sym.hybridize.return_value = None
    dummy_sym.collect_params.return_value = mock.MagicMock()
    dummy_sym.collect_params.return_value.load.return_value = None

    mocker.patch("mxnet.gluon.SymbolBlock").return_value = dummy_sym
    mocker.patch(
        "lorien.dialect.tvm_dial.frontend_parser.relay.frontend.from_mxnet"
    ).return_value = ("mod", "params")

    configs = argparse.Namespace(target=["llvm"], mxnet=["alexnet: { data: [1, 3, 224, 224]}"])

    mod_n_params = EXTRACTOR_FUNC_TABLE["mxnet"](configs)
    assert len(mod_n_params) == 1
    assert mod_n_params[0][0] == "alexnet"
    assert mod_n_params[0][1][0] == "mod"
    assert mod_n_params[0][1][1] == "params"

    mocker.patch("mxnet.gluon.SymbolBlock").side_effect = Exception("Mocked Error")
    mod_n_params = EXTRACTOR_FUNC_TABLE["mxnet"](configs)
    assert len(mod_n_params) == 0
