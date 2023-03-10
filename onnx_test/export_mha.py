import torch
import torch.onnx as onnx
import numpy as np
import onnxruntime
import numpy as np
import os

module_input_shapes = {}
module_name_to_module = {}
# use hook to trace different module input output shape
def input_hook(module, input, output):
    # print('input shape: ', input[0].shape)
    module_input_shapes[module.name] = input[0].shape
    module_name_to_module[module.name] = module

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
    
        # register forward hooks
        self.query.register_forward_hook(input_hook)
        self.key.register_forward_hook(input_hook)
        self.value.register_forward_hook(input_hook)
        self.softmax.register_forward_hook(input_hook)
        self.fc.register_forward_hook(input_hook)
        # assign name to each module
        self.query.name = 'query'
        self.key.name = 'key'
        self.value.name = 'value'
        self.softmax.name = 'softmax'
        self.fc.name = 'fc'


        
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

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=256) # 256 or 512 or 1025
    parser.add_argument('--num_heads', type=int, default=8) # 8 or 16 
    parser.add_argument('--batch_size', type=int, default=2) # 1-32
    parser.add_argument('--seq_length', type=int, default=128) # 128-512
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    hidden_size = args.hidden_size
    num_heads = args.num_heads
    batch_size = args.batch_size
    seq_length = args.seq_length
    input_shape = (batch_size, seq_length, hidden_size)
    # Define an instance of the MHA module
    mha = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
    # input into the module
    x = torch.randn(input_shape)
    # run the module
    y = mha(x)

    # based on the input, export all the modules in the model
    # make a folder to store onnx models
    if not os.path.exists('onnx_models'):
        os.mkdir('onnx_models')
    # export all models in the module_input_shapes to its onnx form
    for module_name, input_shape in module_input_shapes.items():
        input_names = [module_name + '_input']
        output_names = [module_name + '_output']
        module = module_name_to_module[module_name]
        onnx_path = os.path.join('onnx_models', module_name + '.onnx')
        # set the first dim dynamic 
        # dynamic_axes = {input_names[0]: {0: 'batch_size', 1: 'seq_length'}, output_names[0]: {0: 'batch_size', 1: 'seq_length'}}
        torch.onnx.export(module, torch.randn(input_shape), onnx_path, input_names=input_names, output_names=output_names)
# input_names = ['input']
# output_names = ['output']
# dynamic_axes = {'input': {0: 'batch_size', 1: 'seq_length'}, 'output': {0: 'batch_size', 1: 'seq_length'}}
# onnx_path = 'mha.onnx'
# torch.onnx.export(mha, torch.randn(input_shape), onnx_path, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)

# 核心贡献
# - 变长调度器
# - 1. Identify the performance gap in branch performance selection
# - 1. Scheduler
#     - based on the dynamic of the input, dynamically adjust the result
#     - Especially in decoding, with increasing KV length, the performance of the branch will be different
# - 2. Intra batching and core scheduling
