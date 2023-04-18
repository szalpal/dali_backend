# The MIT License (MIT)
#
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import json

import torch
import triton_python_backend_utils as pb_utils
from torchaudio.io import StreamReader
from torchaudio.utils import ffmpeg_utils
import time
import numpy as np
import tempfile

print("Library versions:")
print(ffmpeg_utils.get_versions())
print("\nBuild config:")
print(ffmpeg_utils.get_build_config())
print("\nDecoders:")
print([k for k in ffmpeg_utils.get_video_decoders().keys() if "cuvid" in k])
print("\nEncoders:")
print([k for k in ffmpeg_utils.get_video_encoders().keys() if "nvenc" in k])

cuda_conf = {
    "decoder": "h264_cuvid",  # Use CUDA HW decoder
    "hw_accel": "cuda:0",  # Then keep the memory on CUDA:0
}

FRAMES_PER_SEQUENCE = 5


class TritonPythonModel:

    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])

        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            model_config)
        if not using_decoupled:
            raise pb_utils.TritonModelException(
                """the model `{}` can generate any number of responses per request,
                enable decoupled transaction policy in model configuration to
                serve this model""".format(args['model_name']))

        # Get OUT configuration
        out_config = pb_utils.get_output_config_by_name(model_config, "PYTHON_OUTPUT_0")

        # Convert Triton types to numpy types
        self.out_dtype = pb_utils.triton_string_to_numpy(out_config['data_type'])

    def execute(self, requests):
        # This model does not support batching, so 'request_count' should always be 1.
        if len(requests) != 1:
            raise pb_utils.TritonModelException("unsupported batch size " + len(requests))

        response_sender = requests[0].get_response_sender()

        in_input = pb_utils.get_input_tensor_by_name(requests[0], 'PYTHON_INPUT_0').as_numpy()

        with tempfile.NamedTemporaryFile() as fp:
            in_input.tofile(fp)

            s = StreamReader(fp.name)
            s.add_video_stream(FRAMES_PER_SEQUENCE, **cuda_conf)

            for (chunk,) in s.stream():
                out0_tensor = pb_utils.Tensor.from_dlpack("PYTHON_OUTPUT_0", torch.utils.dlpack.to_dlpack(chunk))
                response = pb_utils.InferenceResponse(output_tensors=[out0_tensor])
                response_sender.send(response)

        response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

        return None

    def finalize(self):
        pass
