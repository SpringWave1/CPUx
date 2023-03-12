# Benchmark includes
# BERT
    # on SST2, WNLI
# ResNet50
import datasets
import onnx
import torch
from transformers import AutoTokenizer
from transformers import BertModel
import numpy as np 
import os 
from time import perf_counter
import pytest
import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm import relay, autotvm
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_executor as runtime
import tempfile

def load_data_lang():
    sst2 = datasets.load_dataset('glue', 'sst2')
    wnli = datasets.load_dataset('glue', 'wnli')
    return sst2, wnli

def load_data_classification():
    cifar10 = datasets.load_dataset('cifar10')
    return cifar10






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


# TVM TUNNING
def tune_kernels(
    tasks, measure_option, tuner="gridsearch", early_stopping=None, log_filename="tuning.log"
):

    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(task, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(task)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if os.path.isfile(tmp_log):
            tuner_obj.load_history(autotvm.record.load_from_file(tmp_log))
        # do tuning
        n_trial = len(task.config_space)
        with tempfile.NamedTemporaryFile() as tmp_task_log_file:    
            tuner_obj.tune(
                n_trial=n_trial,
                early_stopping=early_stopping,
                measure_option=measure_option,
                callbacks=[
                    autotvm.callback.progress_bar(n_trial, prefix=prefix),
                    autotvm.callback.log_to_file(log_filename),
                ],
            )
            with open(tmp_log, 'a') as tmp_log_file:
                tmp_log_file.write(tmp_task_log_file.read().decode('utf8'))
    autotvm.record.pick_best(tmp_log, log)
    os.remove(tmp_log)

def tune_graph(graph, dshape, records, opt_sch_file, use_DP=True):
    target_op = [
        relay.op.get("nn.linear"),
    ]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {input_name: dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)


def evaluate_performance(lib, data_shape):
    # upload parameters to device
    dev = tvm.cpu()
    data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
    module = runtime.GraphModule(lib["default"](dev))
    module.set_input(input_name, data_tvm)

    # evaluate
    print("Evaluate inference time cost...")
    print(module.benchmark(dev, number=100, repeat=3))


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
    mod, params = relay.frontend.from_pytorch(script_module, input_shapes)

    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params)

    # run tuning tasks
    tune_kernels(tasks, **tuning_option)
    tune_graph(mod["main"], data_shape, log_file, graph_opt_sch_file)

    # target = "llvm -mcpu=cascadelake"
    # target = 'llvm -mcpu=amdgcn-amd-amdhsa' # amd 
    # target = tvm.target.Target("llvm", host="llvm")dir
    # target = "llvm -mcpu=skylake-avx512 -libs=cblas"
    # dev = tvm.cpu(0)
    # with tvm.transform.PassContext(opt_level=3):
    # # required_pass=["FastMath"] use approximation
    # # with tvm.transform.PassContext(opt_level=3, required_pass=["FastMath"]):
    #     lib = relay.build(mod, target=target, params=params)
    
    # rt = graph_executor.GraphModule(lib["default"](dev))

    # compile kernels in default mode
    print("Evaluation of the network compiled in 'default' mode without auto tune:")
    with tvm.transform.PassContext(opt_level=3):
        print("Compile...")
        lib = relay.build(mod, target=target, params=params)
        evaluate_performance(lib, data_shape)

    # compile kernels in kernel tuned only mode
    print("\nEvaluation of the network been tuned on kernel level:")
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
        evaluate_performance(lib, data_shape)

    # compile kernels with graph-level best records
    print("\nEvaluation of the network been tuned on graph level:")
    with autotvm.apply_graph_best(graph_opt_sch_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)
        evaluate_performance(lib, data_shape)
    return rt


if __name__ == '__main__':
    model_name = "bert-base-uncased"
    target = "llvm -mcpu=skylake-avx512"

    # TVM TUNNING
    log_file = "%s.log" % model_name
    tmp_log = "%s.tmp.log" % model_name
    graph_opt_sch_file = "%s_graph_opt.log" % model_name
    tuning_option = {
    "log_filename": log_file,
    "tuner": "random",
    "early_stopping": None,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(
            number=1, repeat=10, min_repeat_ms=0, enable_cpu_cache_flush=True
        ),
        ),
    }


    sst2, wnli = load_data_lang()
    cifar10_data = load_data_classification()
    count_times = 10

    bs = 16
    max_seq_length = 128
    
    intra_op_num_threads = 8
    torch.set_num_threads(intra_op_num_threads)
    # BERT
    onnx_model_path = '3rd_baseline/bert-base-uncased.onnx'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    sst_vali = sst2['validation']['sentence']
    wnli_vali = wnli['validation']['sentence1']
    
    sst_vali_processed = preprocessing_dps(tokenizer, bs, sst_vali)
    wnli_vali_processed = preprocessing_dps(tokenizer, bs, wnli_vali)
    tvm_rt = tvm_rt_bert()

    
    print("Evaluate inference time cost...")
    print(tvm_rt.benchmark(dev, number=count_times, repeat=3))
    