# load onnx files from the onnx_models folder
import onnx
import os 
import onnxruntime
from onnxruntime import SessionOptions, InferenceSession
import numpy as np
from time import perf_counter
# import multi thead
from concurrent.futures import ThreadPoolExecutor
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--seq_length', type=int, default=128)
    # cpu sum
    parser.add_argument('--cpu_each', type=int, default=3)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # try simple branching
    input_shape = args.batch_size, args.seq_length, args.hidden_size
    input_data = np.random.random(size=input_shape).astype(np.float32)
    cpu_each = args.cpu_each
    # start from 2, generate affinities for each QKV
    idx = 2
    seq_affinities = [i for i in range(2, 2+cpu_each * 3)]
    affinities_dict = {}
    target_layes = ['query', 'key', 'value', 'softmax', 'fc']
    for layer in target_layes:
        affinities_dict[layer] = []
        for i in range(cpu_each):
            affinities_dict[layer].append(str(idx))
            idx += 1
    # print(affinities_dict)
    # generate cpu configs
    cpu_configs = {}
    for layer in target_layes:
        cpu_configs[f'{layer}'] = {'intra_op_num_threads': len( affinities_dict[layer]) + 1, 'affinities': ';'.join([str(i) for i in affinities_dict[layer]]) }
    # print(cpu_configs)
    cpu_seq_configs = {}
    for layer in target_layes:
        cpu_seq_configs[f'{layer}'] = {'intra_op_num_threads': len(seq_affinities) + 1, 'affinities': ';'.join([str(i) for i in seq_affinities])}
    # generate onnx models
    # import pdb; pdb.set_trace()
    model_name_to_onnx_parallel = {}
    model_name_to_onnx_seq = {}
    for f in os.listdir('onnx_models'):
        if f.endswith('.onnx'):
            path = os.path.join('onnx_models', f)
            # model = onnx.load(path)
            # create session for each
            model_name = f.split('.')[0]
            sess_opt = SessionOptions()
            # sess_opt.execution_mode  = ExecutionMode.ORT_PARALLEL 
            intra_op_num_threads = cpu_configs[model_name]['intra_op_num_threads']
            affinities = cpu_configs[model_name]['affinities']
            sess_opt.intra_op_num_threads = intra_op_num_threads
            sess_opt.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            sess_opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Number of affinities equal to thread_pool_size minus one
            sess_opt.add_session_config_entry('session.intra_op_thread_affinities', affinities)
            session = InferenceSession(path, sess_opt, providers=['CPUExecutionProvider'])
            # model_name
            model_name_to_onnx_parallel[model_name] = session
            # create sequential session as well
            sess_opt_seq = SessionOptions()
            # sess_opt.execution_mode  = ExecutionMode.ORT_PARALLEL
            intra_op_num_threads = cpu_seq_configs[model_name]['intra_op_num_threads']
            affinities = cpu_seq_configs[model_name]['affinities']
            sess_opt_seq.intra_op_num_threads = intra_op_num_threads
            sess_opt_seq.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            sess_opt_seq.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Number of affinities equal to thread_pool_size minus one
            sess_opt_seq.add_session_config_entry('session.intra_op_thread_affinities', affinities)
            session_seq = InferenceSession(path, sess_opt_seq, providers=['CPUExecutionProvider'])
            model_name_to_onnx_seq[model_name] = session_seq
            # get inputnames
            input_names = [input.name for input in session.get_inputs()]
            # get outputnames
            output_names = [output.name for output in session.get_outputs()]

    collect_times = 100
    # get sequential sessions
    model_name_to_onnx = model_name_to_onnx_seq
    # excute the model with ort
    q_sess, k_sess, v_sess = model_name_to_onnx['query'], model_name_to_onnx['key'], model_name_to_onnx['value']
    softmax_sess, fc_sess = model_name_to_onnx['softmax'], model_name_to_onnx['fc']
    q_out = q_sess.run(None, {q_sess.get_inputs()[0].name: input_data})
    k_out = k_sess.run(None, {k_sess.get_inputs()[0].name: input_data})
    v_out = v_sess.run(None, {v_sess.get_inputs()[0].name: input_data})
    # seq execution
    start = perf_counter()
    for _ in range(collect_times):
        q_out = q_sess.run(None, {q_sess.get_inputs()[0].name: input_data})
        k_out = k_sess.run(None, {k_sess.get_inputs()[0].name: input_data})
        v_out = v_sess.run(None, {v_sess.get_inputs()[0].name: input_data})
    end = perf_counter()
    print('qkv time (seq): ', end - start)

    # get sequential sessions
    model_name_to_onnx = model_name_to_onnx_parallel
    # excute the model with ort
    q_sess, k_sess, v_sess = model_name_to_onnx['query'], model_name_to_onnx['key'], model_name_to_onnx['value']
    softmax_sess, fc_sess = model_name_to_onnx['softmax'], model_name_to_onnx['fc']
    q_out = q_sess.run(None, {q_sess.get_inputs()[0].name: input_data})
    k_out = k_sess.run(None, {k_sess.get_inputs()[0].name: input_data})
    v_out = v_sess.run(None, {v_sess.get_inputs()[0].name: input_data})
    with ThreadPoolExecutor(max_workers=3) as executor:
        start = perf_counter()
        for _ in range(collect_times):
            q_out = executor.submit(q_sess.run, None, {q_sess.get_inputs()[0].name: input_data})
            k_out = executor.submit(k_sess.run, None, {k_sess.get_inputs()[0].name: input_data})
            v_out = executor.submit(v_sess.run, None, {v_sess.get_inputs()[0].name: input_data})
            # aggregate the results
            futures = [q_out, k_out, v_out]
            # wait for all threads to finish
            for future in futures:
                future.result()

        end = perf_counter()
        print('qkv time (thread): ', end - start)