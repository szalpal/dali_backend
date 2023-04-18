ARG TRITON_VERSION=23.03
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3
FROM ${BASE_IMAGE} as builder

RUN apt update && apt install -y ffmpeg

RUN pip3 install ipdb
RUN pip3 install --progress-bar off --pre torch torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu117 2> /dev/null
RUN pip3 install --progress-bar off boto3 2> /dev/null



