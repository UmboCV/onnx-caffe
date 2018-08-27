#!/usr/bin/env python
from __future__ import print_function

import caffe
import onnx.backend.base
from caffe.proto import caffe_pb2

caffe.set_mode_cpu()
from ._transformers import ConvAddFuser, ConstantsToInitializers
from ._graph import Graph

from . import _operators as cvt
from . import _weightloader as wlr
from ._error_utils import ErrorHandling
from onnx import shape_inference

transformers = [
    ConstantsToInitializers(),
    ConvAddFuser(),
]


class CaffeBackendRep(onnx.backend.base.BackendRep):
    @classmethod
    def __init__(self, net):
        self._net = net

    @classmethod
    def run(self, input):
        for k, v in input.items():
            self._net.blobs[k].data[...] = v
        output = self._net.forward()
        return [v for v in output.values()]


class CaffeBackend(onnx.backend.base.Backend):
    @classmethod
    def prepare(self, model, prototxt=None):
        model = shape_inference.infer_shapes(model)

        if prototxt:
            net = CaffeBackend.onnx_graph_to_caffemodel(model.graph, prototxt)
        else:
            import tempfile
            with tempfile.NamedTemporaryFile() as f:
                net = CaffeBackend.onnx_graph_to_caffemodel(model.graph, f.name)

        return CaffeBackendRep(net)

    @staticmethod
    def onnx_graph_to_caffemodel(model_graph, prototxt_save_path=None):
        graph = Graph.from_onnx(model_graph)
        graph = graph.transformed(transformers)
        graph.channel_dims = {}

        exist_edges = []
        layers = []
        exist_nodes = []
        err = ErrorHandling()
        for i in graph.inputs:
            edge_name = i[0]
            input_layer = cvt.make_input(i)
            layers.append(input_layer)
            exist_edges.append(i[0])
            graph.channel_dims[edge_name] = graph.shape_dict[edge_name][1]

        for id, node in enumerate(graph.nodes):
            node_name = node.name
            op_type = node.op_type
            inputs = node.inputs
            inputs_tensor = node.input_tensors
            input_non_exist_flag = False

            for inp in inputs:
                if inp not in exist_edges and inp not in inputs_tensor:
                    input_non_exist_flag = True
                    break
            if input_non_exist_flag:
                continue

            if op_type not in cvt._ONNX_NODE_REGISTRY:
                err.unsupported_op(node)
                continue
            converter_fn = cvt._ONNX_NODE_REGISTRY[op_type]
            layer = converter_fn(node, graph, err)
            if type(layer) == tuple:
                for l in layer:
                    layers.append(l)
            else:
                layers.append(layer)
            outs = node.outputs
            for out in outs:
                exist_edges.append(out)

        net = caffe_pb2.NetParameter()
        for id, layer in enumerate(layers):
            layers[id] = layer._to_proto()
        net.layer.extend(layers)

        with open(prototxt_save_path, 'w') as f:
            print(net, file=f)

        caffe.set_mode_cpu()
        deploy = prototxt_save_path
        net = caffe.Net(deploy,
                        caffe.TEST)

        for id, node in enumerate(graph.nodes):
            node_name = node.name
            op_type = node.op_type
            inputs = node.inputs
            inputs_tensor = node.input_tensors
            input_non_exist_flag = False
            if op_type not in wlr._ONNX_NODE_REGISTRY:
                err.unsupported_op(node)
                continue
            converter_fn = wlr._ONNX_NODE_REGISTRY[op_type]
            converter_fn(net, node, graph, err)

        # net.save(caffe_model_save_path)
        return net


backend = CaffeBackend()
