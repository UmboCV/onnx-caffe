import numpy as np
import onnx
import pytest
import tempfile
import torch
import torchvision


def _get_rnd(shape, min_value=-10, max_value=10):
    return ((max_value - min_value) * np.random.rand(*shape) + min_value).astype(np.float32)


def build_nodes_ref(model):
    nodes_by_input = {}
    nodes_by_output = {}
    for i, node in enumerate(model.graph.node):
        for x in node.input:
            if x in nodes_by_input:
                nodes_by_input[x].append(node)
            else:
                nodes_by_input[x] = [node]

        for x in node.output:
            if x in nodes_by_output:
                nodes_by_output[x].append(node)
            else:
                nodes_by_output[x] = [node]

    return nodes_by_input, nodes_by_output


def remove_trivail_node(_model):
    # XXX: model clone
    model = _model
    nodes_by_input, _ = build_nodes_ref(model)
    for i, node in enumerate(model.graph.node):
        if node.op_type == 'Pad':
            pads = node.attribute[1]
            assert pads.name == "pads"

            if sum(map(abs, pads.ints)) == 0:
                print('removing Pad')
                for next_node in nodes_by_input[node.output[0]]:
                    next_node.input[[x for _, x in enumerate(next_node.input)].index(node.output[0])] = node.input[0]
    return model


def fold_constant_shape(model):
    # refer to https://github.com/onnx/onnx/blob/9b874e973034c92eafba5369cede3d18e934189e/onnx/onnx.proto#L224
    # for GraphProto
    inferred_model = onnx.shape_inference.infer_shapes(model)
    print('initializer', [x.name for i, x in enumerate(model.graph.initializer)])
    print('input nodes', [x.name for i, x in enumerate(model.graph.input)])
    print('value_info nodes', [x.name for i, x in enumerate(inferred_model.graph.value_info)])
    print('output nodes', [x.name for i, x in enumerate(model.graph.output)])

    shape_info = {}
    for i, info in enumerate(inferred_model.graph.value_info):
        shape_info[info.name] = info.type
        assert int(info.name) or True, "name not an integer"

    for i, node in enumerate(inferred_model.graph.node):
        if node.op_type == 'Shape':
            if node.input[0] in shape_info:
                shape = shape_info[node.input[0]]
                print(f'node {i} Shape', shape)
                # copy inferred shape to become an initializer
                # https://developers.google.com/protocol-buffers/docs/pythontutorial
                init = inferred_model.graph.initializer.add()
                init.name = 'infer_' + node.output[0]
                init.dims.append(4)
                init.data_type = init.DataType.Value('INT64')
                init.int64_data.extend([x.dim_value for i, x in enumerate(shape.tensor_type.shape.dim)])
                node.input[0] = init.name

                inp = inferred_model.graph.input.add()
                inp.name = init.name
                inp.type.CopyFrom(shape)

        if node.op_type == 'Reshape':
            # this is hard coded not inferred
            shape_input_name = node.input[1]
            shape = shape_info[shape_input_name]

            init = inferred_model.graph.initializer.add()
            init.name = 'infer_' + shape_input_name
            init.dims.append(2)
            init.data_type = init.DataType.Value('INT64')
            init.int64_data.extend(
                [x.dim_value for i, x in enumerate(shape_info[node.input[0]].tensor_type.shape.dim)][:2])
            node.input[1] = init.name

            inp = inferred_model.graph.input.add()
            inp.name = init.name
            inp.type.CopyFrom(shape)

    return inferred_model


def remove_unreachable_nodes(_model):
    # XXX model clone
    model = _model
    _, nodes_by_output = build_nodes_ref(model)

    def encode(node):
        return node.op_type + "_".join(node.input)

    reached_node = set()
    reached_values = set()
    queue = [x.name for i, x in enumerate(model.graph.output)]
    while len(queue) > 0:
        adj_nodes = nodes_by_output[queue.pop(0)]
        for node in adj_nodes:
            if encode(node) not in reached_node:
                reached_node.add(encode(node))
                for inp in node.input:
                    reached_values.add(inp)
                    if inp in nodes_by_output:
                        queue.append(inp)
                for inp in node.output:
                    reached_values.add(inp)

    remove_nodes = []
    for i, node in enumerate(model.graph.node):
        if encode(node) not in reached_node:
            print("to prune node " + encode(node))
            remove_nodes.append(node)

    for node in remove_nodes:
        print('removing ' + encode(node))
        model.graph.node.remove(node)

    remove_values = []
    for i, vi in enumerate(model.graph.value_info):
        if vi.name not in reached_values:
            remove_values.append(vi)
            print('to prune value ', vi.name)
    for vi in remove_values:
        model.graph.value_info.remove(vi)

    return model


def optimize_model(model):
    model = remove_trivail_node(model)
    model = fold_constant_shape(model)
    model = remove_unreachable_nodes(model)
    return model


def simple(sz=5):
    torch.random.manual_seed(1)  # fix random weight initalization
    dummy_input = (lambda w: torch.arange(w * w, requires_grad=False).reshape(1, 1, w, w).float())(sz)
    torch_model = torch.nn.Sequential(
        torch.nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1),
        torch.nn.MaxPool2d(kernel_size=3, padding=[1, 1], stride=[2, 2])
    )

    onnx_model_path = tempfile.mktemp()
    with torch.no_grad():
        torch_out = torch.onnx._export(torch_model,  # model being run
                                       dummy_input,  # model input (or a tuple for multiple inputs)
                                       onnx_model_path,  # where to save the model (can be a file or file-like object)
                                       export_params=True)  # store the trained parameter weights inside the model file
    # Load the ONNX ModelProto object. model is a standard Python protobuf object
    model = onnx.load(onnx_model_path)
    kwinput = {model.graph.input[0].name: dummy_input.numpy()}
    output = torch_out.numpy()
    return model, kwinput, output


def resnet50():
    onnx_model_path = tempfile.mktemp()
    dummy_input = torch.rand(1, 3, 224, 224, requires_grad=False)
    torch_model = torchvision.models.resnet50(pretrained=False)
    torch_model.train(False)
    with torch.no_grad():
        torch_out = torch.onnx._export(torch_model,  # model being run
                                       dummy_input,  # model input (or a tuple for multiple inputs)
                                       onnx_model_path,  # where to save the model (can be a file or file-like object)
                                       export_params=True)  # store the trained parameter weights inside the model file
    # Load the ONNX ModelProto object. model is a standard Python protobuf object
    model = onnx.load(onnx_model_path)
    kwinput = {model.graph.input[0].name: dummy_input.numpy()}
    output = torch_out.numpy()
    return model, kwinput, output


def optimized_resnet50():
    _model, _input, _output = resnet50()
    _model = optimize_model(_model)
    return _model, _input, _output


@pytest.fixture(params=[
    'simple(4)',
    'simple(5)',
    'simple(6)',
    'optimized_resnet50()'
])
def testset(request):
    return eval(request.param)
