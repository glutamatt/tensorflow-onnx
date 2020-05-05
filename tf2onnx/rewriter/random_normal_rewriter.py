# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter - rewrite tensorflow subgraph to onnx random normal op
"""

from tf2onnx import utils
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher
from tf2onnx import constants, logging

logger = logging.getLogger(__name__)

# pylint: disable=missing-docstring


def rewrite_random_normal(g, ops):
    pattern = \
        OpTypePattern('Add', name='output', inputs=[
            OpTypePattern('Mul', name='input2', inputs=[
                OpTypePattern('RandomStandardNormal', name='input1', inputs=["*"]), "*"
            ]), "*"
        ])

    matcher = GraphMatcher(pattern)
    match_results = list(matcher.match_ops(ops))
    logger.info("===========RandomStandardNormal===============")
    for match in match_results:
        output = match.get_op('output')
        mean = output.inputs[1].get_tensor_value()
        dtype = g.get_dtype(output.output[0])
        op_name = utils.make_name("RandomNormal")
        out_name = utils.port_name(op_name)

        rn_op = match.get_op('input1')
        seed = rn_op.get_attr('seed2').i
        if rn_op.inputs[0].type == "Shape":
            shape_node = rn_op.inputs[0]
            print("make_node RandomNormalLike")
            logger.info("make_node RandomNormalLike")
            new_node = g.make_node("RandomNormalLike", [shape_node.input[0]], outputs=[out_name], name=op_name,
                                   attr={"mean": mean, "scale": 1.0, "dtype": dtype, "seed": seed})
        else:
            shape = g.get_shape(output.output[0])
            print("make_node RandomNormal")
            logger.info("make_node RandomNormalLike")
            new_node = g.make_node("RandomNormal", [], outputs=[out_name], name=op_name,
                                   attr={"shape": shape, "mean": mean, "scale": 1.0, "dtype": dtype, "seed": seed})

        g.replace_all_inputs(ops, output.output[0], new_node.output[0])
        g.safe_remove_nodes(match.get_nodes())
    return ops
