import torch
import torch.onnx as onnx
import numpy as np

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.query = torch.nn.Linear(hidden_size, hidden_size)
        self.key = torch.nn.Linear(hidden_size, hidden_size)
        self.value = torch.nn.Linear(hidden_size, hidden_size)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.fc = torch.nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        q = torch.cat(torch.split(q, self.hidden_size // self.num_heads, dim=-1), dim=0)
        k = torch.cat(torch.split(k, self.hidden_size // self.num_heads, dim=-1), dim=0)
        v = torch.cat(torch.split(v, self.hidden_size // self.num_heads, dim=-1), dim=0)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.hidden_size // self.num_heads)
        weights = self.softmax(scores)
        attn = torch.matmul(weights, v)
        
        attn = torch.cat(torch.split(attn, x.size(0), dim=0), dim=-1)
        out = self.fc(attn)
        
        return out

# Define an instance of the MHA module
mha = MultiHeadAttention(hidden_size=256, num_heads=8)

# Export the module to ONNX format
input_shape = (10, 128, 256)
input_names = ['input']
output_names = ['output']
dynamic_axes = {'input': {0: 'batch_size', 1: 'seq_length'}, 'output': {0: 'batch_size', 1: 'seq_length'}}
onnx_path = 'mha.onnx'
torch.onnx.export(mha, torch.randn(input_shape), onnx_path, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)

核心贡献
- 变长调度器
- 1. Identify the performance gap in branch performance selection
- 1. Scheduler
    - based on the dynamic of the input, dynamically adjust the result
    - Especially in decoding, with increasing KV length, the performance of the branch will be different
- 2. Intra batching and core scheduling
