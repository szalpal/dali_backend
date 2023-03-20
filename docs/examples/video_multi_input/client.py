#!/usr/bin/env python

# The MIT License (MIT)
#
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES
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

import argparse
import os
import sys
import numpy as np
import tritonclient.grpc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')
    parser.add_argument('--video', type=str, required=False, default=None,
                        help='Path to a directory, where the video data is located.')
    return parser.parse_args()


def handle_dali_result(result, error):
    if error is not None:
        print(error)
    else:
        # Get the output and show the result
        output0_data = result.as_numpy("OUTPUT_0")
        print(output0_data.shape)

        output1_data = result.as_numpy("OUTPUT_1")
        print(output1_data.shape)

        output2_data = result.as_numpy("OUTPUT_2")
        print(output2_data.shape)
        # Do something with the output...


def load_videos(filenames):
    """
    Loads all files in given directory path. Treats them as binary files.
    """
    return [np.fromfile(filename, dtype=np.uint8) for filename in filenames]


def array_from_list(arrays):
    """
    Convert list of ndarrays to single ndarray with ndims+=1. Pad if necessary.
    """
    lengths = list(map(lambda x, arr=arrays: arr[x].shape[0], [x for x in range(len(arrays))]))
    max_len = max(lengths)
    arrays = list(map(lambda arr, ml=max_len: np.pad(arr, (0, ml - arr.shape[0])), arrays))
    for arr in arrays:
        assert arr.shape == arrays[0].shape, "Arrays must have the same shape"
    return np.stack(arrays)


def main():
    FLAGS = parse_args()

    input_batch_size = 1  # `fn.inputs.video` accepts only one sample at the input,
    max_batch_size = 3    # but the output from `fn.inputs.video` has always max_batch_size.

    # Load test data
    if FLAGS.video is None:
        dali_extra_path = os.environ['DALI_EXTRA_PATH']
        filenames = [
            os.path.join(dali_extra_path, "db", "video", "containers", "mkv", "cfr.mkv"),
            os.path.join(dali_extra_path, "db", "video", "containers", "mkv", "cfr.mkv"),
            os.path.join(dali_extra_path, "db", "video", "containers", "mkv", "cfr.mkv"),
        ]
    else:
        filenames = [os.path.join(FLAGS.video, p) for p in os.listdir(FLAGS.video)]
        filenames = filenames[:input_batch_size]

    try:
        triton_client = tritonclient.grpc.InferenceServerClient(url=FLAGS.url)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    model_name = "model.dali"
    model_version = -1

    # Start stream
    triton_client.start_stream(callback=handle_dali_result)

    # Config output
    outputs = []
    output_names = [
        "OUTPUT_0",
        "OUTPUT_1",
        "OUTPUT_2"
    ]
    for on in output_names:
        outputs.append(tritonclient.grpc.InferRequestedOutput(on))

    # Config inputs: 3 video files, one per every input
    video_raw = load_videos(filenames)
    video_raw = array_from_list(video_raw)
    input_shape = list(video_raw.shape)

    # Initialize tritonserver inputs
    inputs = [
        tritonclient.grpc.InferInput("INPUT_0", input_shape, "UINT8"),
        tritonclient.grpc.InferInput("INPUT_1", input_shape, "UINT8"),
        tritonclient.grpc.InferInput("INPUT_2", input_shape, "UINT8"),
    ]

    inputs[0].set_data_from_numpy(video_raw)
    inputs[1].set_data_from_numpy(video_raw)
    inputs[2].set_data_from_numpy(video_raw)

    request_id = "0"
    triton_client.async_stream_infer(model_name=model_name, inputs=inputs, request_id=request_id,
                                     outputs=outputs)


if __name__ == '__main__':
    main()
