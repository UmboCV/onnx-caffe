#!/usr/bin/env python
from __future__ import absolute_import
# pytorch -> onnx --> caffe --> (pb2, prototxt) file

import numpy as np
import onnx
import os
import pytest

from common import testset, simple

# ONNX generic


@pytest.fixture(params=['caffe'])
def backend(request):
    from onnx_caffe import backend as caffe_backend
    return caffe_backend

def test_caffes_result(backend, testset):
    _model, _input, _output = testset
    prepared_backend = backend.prepare(_model)

    # Run the net:
    output = prepared_backend.run(_input)[0]
    np.testing.assert_almost_equal(_output, output, decimal=3)
    print(_output, output)
    print("Exported model has been executed on Caffe2 backend, and the result looks good!")


def test_caffe_save_model(testset, request):
    model_dir = '.caffe_model/' + request.node.name + '/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Apply the optimization on the original model
    onnx_model, _, _ = testset
    with open(model_dir + '/optimized_model.onnx', 'wb') as f:
        onnx.save(onnx_model, f)
    with open(model_dir + '/optimized_model.onnx.txt', 'w') as f:
        f.write(onnx.helper.printable_graph(onnx_model.graph))
        for i, v in enumerate(onnx_model.graph.value_info):
            f.write("{}\n{}\n----".format(i, v))

    from onnx_caffe import CaffeBackend
    caffemodel = CaffeBackend.onnx_graph_to_caffemodel(onnx_model.graph,
                                                       prototxt_save_path=model_dir + '/optimized_model.prototxt')
    caffemodel.save(model_dir + '/optimized_model.caffemodel')
