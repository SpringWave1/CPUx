BATCHSIZE=4
NUM_HEADS=16
HIDDEN_SIZE=256
SEQ_LENGTH=256
CPU_EACH=4
python3 export_mha.py --batch_size $BATCHSIZE --num_heads $NUM_HEADS --hidden_size $HIDDEN_SIZE --seq_length $SEQ_LENGTH
python3 onnx_perf.py --batch_size $BATCHSIZE --num_heads $NUM_HEADS --hidden_size $HIDDEN_SIZE --seq_length $SEQ_LENGTH --cpu_each $CPU_EACH