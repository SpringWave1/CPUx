import torch
import transformers
import turbo_transformers
from transformers import AutoTokenizer
import datasets
import numpy as np 
from time import perf_counter
# bert result preprocessing
def get_max_length_in_array(token_array):
    lengths = [len(i) for i in token_array]
    max_length = max(lengths)
    return max_length, lengths

def pad_to_max_length(token_array, max_length):
    _, lengths = get_max_length_in_array(token_array)
    rows = token_array.shape[1]
    new_token_array = np.zeros((rows, max_length))
    for row_idx in range(rows):
        token_a = token_array[0,row_idx]
        length = len(token_a)
        new_token_array[row_idx, :length] = token_a
    return new_token_array

def data_padding(encoded):
    for k, v in encoded.items():
        encoded[k] = pad_to_max_length(v, max_seq_length)
    
def preprocessing_dps(tokenizer, bs, sst_data):
    # shuffle the data
    np.random.shuffle(sst_data)
    data = sst_data[:bs]
    encoded = dict(tokenizer(data, return_tensors='np'))
    data_padding(encoded)
    return encoded

def load_data_lang():
    sst2 = datasets.load_dataset('glue', 'sst2')
    wnli = datasets.load_dataset('glue', 'wnli')
    return sst2, wnli

if __name__ == "__main__":
    
    model_id = "bert-base-uncased"
    model = transformers.BertModel.from_pretrained(model_id, torchscript=True)
    model.eval()
    cfg = model.config

    bs = 32
    max_seq_length = 128
    
    intra_op_num_threads = 16
    torch.set_num_threads(intra_op_num_threads)
    turbo_transformers.set_num_threads(intra_op_num_threads)

    tt_model = turbo_transformers.BertModel.from_torch(model)


    sst2, wnli = load_data_lang()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    sst_vali = sst2['validation']['sentence']
    wnli_vali = wnli['validation']['sentence1']
    
    sst_vali_processed = preprocessing_dps(tokenizer, bs, sst_vali)
    wnli_vali_processed = preprocessing_dps(tokenizer, bs, wnli_vali)

    input_names = ['input_ids', 'token_type_ids', 'attention_mask']
    torch_value = [torch.tensor(sst_vali_processed[k], dtype=torch.long) for k in input_names[:-1]]

    res = tt_model(*torch_value)
    # input_ids, position_ids=position_ids,
    # token_type_ids=segment_ids)  # pooled_output, sequence_output
    count_times = 10
    start = perf_counter()
    for _ in range(count_times):
        tt_model(*torch_value)
    dur = perf_counter() - start
    print(f'tt inference time: {dur/count_times}')