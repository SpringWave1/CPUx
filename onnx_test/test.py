import onnx 
file_name = 'mha.onnx'
model = onnx.load(file_name)
# print model with node names
print(onnx.helper.printable_graph(model.graph))