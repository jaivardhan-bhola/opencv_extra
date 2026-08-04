import onnx
from onnx import TensorProto, checker, helper


def constant(name, output, data_type, dims, values):
    value = helper.make_tensor(name + "_value", data_type, dims, values)
    return helper.make_node("Constant", [], [output], name=name, value=value)


def make_gather_cast_model(path):
    nodes = [
        helper.make_node("Shape", ["input"], ["shape"], name="shape"),
        constant("index", "index", TensorProto.INT64, [1], [1]),
        helper.make_node("Gather", ["shape", "index"], ["gather"], name="gather", axis=0),
        helper.make_node("Cast", ["gather"], ["gather_float"], name="cast", to=TensorProto.FLOAT),
        constant("exponent", "exponent", TensorProto.FLOAT, [1], [2.0]),
        helper.make_node("Pow", ["gather_float", "exponent"], ["squared"], name="pow"),
        constant("numerator", "numerator", TensorProto.FLOAT, [1], [18.0]),
        helper.make_node("Div", ["numerator", "squared"], ["output"], name="div"),
    ]
    graph = helper.make_graph(
        nodes,
        "gather_cast_float",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(
        graph,
        producer_name="gather-cast-regression",
        opset_imports=[helper.make_operatorsetid("", 16)],
    )
    model.ir_version = 9
    checker.check_model(model)
    onnx.save(model, path)


make_gather_cast_model("models/gather_cast_float.onnx")
