import json
import os
from pathlib import Path

from transformers import AutoTokenizer
from transformers.onnx import export
from transformers.onnx.features import FeaturesManager

import numpy as np
# current folder
current_path = os.path.dirname(os.path.abspath(__file__))

# export an transformer model to onnx
def export_onnx_transformer(model_name):
    # onnx path
    onnx_path = Path(current_path) / f'{model_name}.onnx'
    # check if the onnx file exists
    # if exists, return
    if onnx_path.exists():
        print(f'{model_name} already exists')
        return
    # Download the model
    transformers_model = FeaturesManager.get_model_from_feature('default', model_name)
    _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(transformers_model, feature='default')
    onnx_config = model_onnx_config(transformers_model.config)
    # Download the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Export .onnx
    try:
        export(tokenizer, transformers_model, onnx_config, onnx_config.default_onnx_opset, onnx_path)
        # add name to config
        add_model(model_name)
        print(f'Export {model_name} success')
    except:
        print(f'Export {model_name} failed')
        return

if __name__ == '__main__':
    model_names = ['bert-base-uncased', 'gpt2']
    for model_name in model_names:
        export_onnx_transformer(model_name)