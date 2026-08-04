import onnx
from onnx import TensorProto, checker, helper


def constant(name, output, data_type, dims, values):
    value = helper.make_tensor(name + "_value", data_type, dims, values)
    return helper.make_node("Constant", [], [output], name=name, value=value)


def make_mul_cast_model(path):
    nodes = [
        helper.make_node("Shape", ["input"], ["shape"], name="shape"),
        constant("scale", "scale", TensorProto.INT64, [2], [2, 2]),
        helper.make_node("Mul", ["shape", "scale"], ["scaled_shape"], name="mul"),
        helper.make_node("Cast", ["scaled_shape"], ["scaled_shape_float"], name="cast", to=TensorProto.FLOAT),
        constant("divisor", "divisor", TensorProto.FLOAT, [2], [2.0, 3.0]),
        helper.make_node("Div", ["scaled_shape_float", "divisor"], ["output"], name="div"),
    ]
    graph = helper.make_graph(
        nodes,
        "mul_cast_float",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])],
    )
    model = helper.make_model(
        graph,
        producer_name="mul-cast-regression",
        opset_imports=[helper.make_operatorsetid("", 16)],
    )
    model.ir_version = 9
    checker.check_model(model)
    onnx.save(model, path)


make_mul_cast_model("models/mul_cast_float.onnx")
