
import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np
import onnxruntime as rt

# write an onnx version of multihead attention
# input: [batch_size, seq_len, hidden_size]
# output: [batch_size, seq_len, hidden_size]
# weights: [hidden_size, hidden_size]
# num_heads: int
def create_onnx_model(input_name, output_name, weights_name, num_heads):
    # Create one input (ValueInfoProto)
    X = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [None, None, None])
    W = helper.make_tensor_value_info(weights_name, TensorProto.FLOAT, [None, None])
    Y = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [None, None, None])

    # Create a node (NodeProto)
    node_def = helper.make_node(
        'MultiHeadAttention',
        inputs=[input_name, weights_name, weights_name, weights_name],
        outputs=[output_name],
        num_heads=num_heads,
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [X, W],
        [Y],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-MHA')
    return model_def

def run_onnx_model(model, input_data, weights_data):
    sess = rt.InferenceSession(model.SerializeToString())
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    res = sess.run([output_name], {input_name: input_data, 'W': weights_data})
    return res
