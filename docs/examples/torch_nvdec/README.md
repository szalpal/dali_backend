# Using NVDEC with Python Backend and Torch

This example shows how to use NVDEC in Triton using Python Backend and Torch.

## How to run
1. Build tritonserver docker image with provided Dockerfile
```
docker build -t tritonserver:torch_nvdec .
```
1. Run the docker image. Remember to fill `$MODEL_REPO` variable properly.
```
docker run -it --rm --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 --privileged -v $MODEL_REPO:/models tritonserver:torch_nvdec tritonserver --model-repository /models --log-verbose 1
```
1. Run the client. Remember to set the path to the input file.
```
python client.py -f input.mp4
``` 

## Remarks
There is a bug in Torch. `StreamReader` is not able to accept in-memory data. Therefore, this example comprises a workaround. In-memory data is saved to a temporary file and then decoded in the `StreamReader`.