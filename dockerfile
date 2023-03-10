# FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

ARG PROJECT_DIR=/workspace

ADD . $PROJECT_DIR
WORKDIR $PROJECT_DIR
ENV HOME $PROJECT_DIR
# change data path and number of trainers here
ENV PYTHONPATH $PROJECT_DIR
RUN apt-get update && apt-get install -y \
    numactl git ffmpeg libsm6 libxext6 ca-certificates curl jq
RUN pip install transformers[onnx] optimum-transformers optimum[exporters] datasets diffusers accelerate\
    psutil protobuf docker grpcio opencv-python pympler fastjsonschema ruptures requests tqdm \
    onnx onnxruntime \
    tensorflow \
    openvino-dev openvino
    
CMD echo "===========END========="
CMD /bin/bash
