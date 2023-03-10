import torch
import torch.nn as nn
import onnx
import onnxruntime
from time import perf_counter
from onnxruntime import SessionOptions, InferenceSession
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 4*hidden_size),
            nn.ReLU(),
            nn.Linear(4*hidden_size, hidden_size),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        residual = x
        x, _ = self.self_attention(x, x, x)
        x = self.dropout1(x)
        x = self.layer_norm1(residual + x)
        residual = x
        x = self.feed_forward(x)
        x = self.dropout2(x)
        x = self.layer_norm2(residual + x)
        return x

hidden_size = 256
num_heads = 8
dropout = 0.1
seq_len = 512
batch_size = 10
CPU_NUMS = 8
# Define an example input to the model
input_shape = (batch_size, seq_len, hidden_size)
input_data = torch.randn(input_shape)

# Create an instance of the model
model = TransformerBlock( hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)

affinities = ";".join([str(i) for i in range(2, 2+CPU_NUMS - 1)])
# Export the model to ONNX
output_path = "transformer_block.onnx"
torch.onnx.export(model, input_data, output_path)

sess_opt = SessionOptions()
sess_opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_opt.intra_op_num_threads = CPU_NUMS
sess_opt.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
sess_opt.add_session_config_entry('session.intra_op_thread_affinities', affinities)

ort_session = onnxruntime.InferenceSession(output_path, sess_opt, providers=['CPUExecutionProvider'])
# get input names
input_names = ort_session.get_inputs()[0].name
ort_inputs = {input_names: input_data.detach().numpy()}
ort_outputs = ort_session.run(None, ort_inputs)
# print(ort_outputs[0])


collect_times = 5
start = perf_counter()
for _ in range(collect_times):
    ort_outputs = ort_session.run(None, ort_inputs)
end = perf_counter()
print("Time: {}".format(end - start))


sess_opt = SessionOptions()
sess_opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_opt.intra_op_num_threads = CPU_NUMS
sess_opt.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
sess_opt.inter_op_num_threads = 3
ort_session = onnxruntime.InferenceSession(output_path, sess_opt, providers=['CPUExecutionProvider'])
sess_opt.add_session_config_entry('session.intra_op_thread_affinities', affinities)

ort_outputs = ort_session.run(None, ort_inputs)
start = perf_counter()
for _ in range(collect_times):
    ort_outputs = ort_session.run(None, ort_inputs)
end = perf_counter()
print("Time: {}".format(end - start))