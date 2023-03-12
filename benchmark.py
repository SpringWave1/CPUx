# Benchmark includes
# BERT
    # on SST2, WNLI
# ResNet50
import datasets
import onnx
import onnxruntime as ort 
from openvino.tools.mo import convert_model
from openvino.runtime import Core, PartialShape
from onnxruntime import SessionOptions, InferenceSession
from transformers import BertModel
import torch
from transformers import AutoTokenizer
import numpy as np 
import os 
import tvm
from tvm import relay
from time import perf_counter
from tvm.contrib import graph_executor

def load_data_lang():
    sst2 = datasets.load_dataset('glue', 'sst2')
    wnli = datasets.load_dataset('glue', 'wnli')
    return sst2, wnli

def load_data_classification():
    cifar10 = datasets.load_dataset('cifar10')
    return cifar10

def create_onnx_rt(onnx_model_path):
    sess_opt = SessionOptions()
    # sess_opt.execution_mode  = ExecutionMode.ORT_PARALLEL 
    sess_opt.intra_op_num_threads = intra_op_num_threads 
    sess_opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = InferenceSession(onnx_model_path, sess_opt, providers=['CPUExecutionProvider'])

    input_names = [i.name for i in sess.get_inputs()]
    output_names = [i.name for i in sess.get_outputs()]

    return sess, input_names, output_names

def create_ov_rt(onnx_model_path):
    ov_model = convert_model(onnx_model_path)
    tput = {'INFERENCE_NUM_THREADS':intra_op_num_threads, 'NUM_STREAMS': 1, "ALLOW_AUTO_BATCHING": False}
    core = Core()
    model = core.compile_model(ov_model, 'CPU', tput) 
    return model

def torch_jit_rt_bert():
    model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)
    model.eval()
    return model 

def tvm_rt_bert():
    model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)
    os.environ["TVM_NUM_THREADS"] = str(intra_op_num_threads)

    batch_size = bs
    seq_len = max_seq_length
    inputs = (torch.ones(batch_size, seq_len, dtype=torch.int64),
            torch.ones(batch_size, seq_len, dtype=torch.int64),
            torch.ones(batch_size, seq_len, dtype=torch.int64))

    input_shapes = [("input_ids", (inputs[0].shape, "int64")),
                ("attention_mask", (inputs[1].shape, "int64")),
                ("token_type_ids", (inputs[2].shape, "int64"))]

    script_module = torch.jit.trace(model, inputs).eval()

    import time
    t1 = time.time()
    mod, params = relay.frontend.from_pytorch(script_module, input_shapes)
    t2 = time.time()

    print(relay.transform.InferType()(mod))

    print("PT import time:", t2 - t1)

    # target = "llvm -mcpu=cascadelake"
    # target = 'llvm -mcpu=amdgcn-amd-amdhsa' # amd 
    # target = tvm.target.Target("llvm", host="llvm")dir
    # target = "llvm -mcpu=skylake-avx512 -libs=cblas"
    target = "llvm"
    dev = tvm.cpu(0)
    with tvm.transform.PassContext(opt_level=3):
    # required_pass=["FastMath"] use approximation
    # with tvm.transform.PassContext(opt_level=3, required_pass=["FastMath"]):
        lib = relay.build(mod, target=target, params=params)
    
    rt = graph_executor.GraphModule(lib["default"](dev))
    return rt




# bert result preprocessing
def get_max_length_in_array(token_array):
    lengths = [len(i) for i in token_array]
    max_length = max(lengths)
    return max_length, lengths

def pad_to_max_length(token_array, max_length):
    _, lengths = get_max_length_in_array(token_array)
    new_token_array = []
    for idx, token_list in enumerate(token_array):
        pad_num = max_length - lengths[idx]
        new_token_array.append(np.pad(token_list, (0, pad_num), 'constant'))
    return new_token_array

def data_padding(encoded):
    for k, v in encoded.items():
        encoded[k] = np.array(pad_to_max_length(v, max_seq_length))
    
def preprocessing_dps(tokenizer, bs, sst_data):
    # shuffle the data
    np.random.shuffle(sst_data)
    data = sst_data[:bs]
    encoded = dict(tokenizer(data, return_tensors='np'))
    data_padding(encoded)
    return encoded



if __name__ == '__main__':
    sst2, wnli = load_data_lang()
    cifar10_data = load_data_classification()

    bs = 16
    max_seq_length = 128
    
    intra_op_num_threads = 8
    torch.set_num_threads(intra_op_num_threads)
    # BERT
    onnx_model_path = '3rd_baseline/bert-base-uncased.onnx'
    onnx_rt, input_names, output_names = create_onnx_rt(onnx_model_path)
    ov_rt = create_ov_rt(onnx_model_path).create_infer_request()
    torch_rt = torch_jit_rt_bert()

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    sst_vali = sst2['validation']['sentence']
    wnli_vali = wnli['validation']['sentence1']
    
    sst_vali_processed = preprocessing_dps(tokenizer, bs, sst_vali)
    wnli_vali_processed = preprocessing_dps(tokenizer, bs, wnli_vali)
    torch_value = [torch.tensor(sst_vali_processed[k]) for k in input_names[:-1]]
    traced_model = torch.jit.trace(torch_rt, torch_value)
    tvm_rt = tvm_rt_bert()

    count_times = 10
    ov_rt.infer(sst_vali_processed)
    start = perf_counter()
    for _ in range(count_times):
        ov_rt.infer(sst_vali_processed)
    dur = perf_counter() - start
    print(f'ov_rt inference time: {dur/count_times}')
    # input_data = {k:sst_vali_processed[k] for k in input_names}
    onnx_rt.run(None, sst_vali_processed)
    start = perf_counter()
    for _ in range(count_times):
        onnx_rt.run(None, sst_vali_processed)
    dur = perf_counter() - start
    print(f'onnx_rt inference time: {dur/count_times}')

    traced_model(*torch_value)
    start = perf_counter()
    for _ in range(count_times):
        traced_model(*torch_value)
    dur = perf_counter() - start
    print(f'torch_rt inference time: {dur/count_times}')

    for k, v in sst_vali_processed.items():
        tvm_rt.set_input(k, v)
    tvm_rt.run()

    start = perf_counter()
    for _ in range(count_times):
        tvm_rt.run()
    dur = perf_counter() - start
    print(f'tvm_rt inference time: {dur/count_times}')
    