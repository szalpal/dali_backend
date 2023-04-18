# The MIT License (MIT)
#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
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


from functools import partial
import queue
import argparse

from tritonclient.utils import *
import tritonclient.grpc as t_client


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


user_data = UserData()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8001',
                        help='Inference server GRPC URL. Default is localhost:8001.')
    parser.add_argument('-f', '--filename', type=str, required=True, help='Path to the test file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model_name = 'torch_nvdec'
    with t_client.InferenceServerClient(url=args.url) as triton_client:
        triton_client.start_stream(callback=partial(callback, user_data))

        input_data = np.fromfile(args.filename, dtype=np.uint8).reshape(1, -1)

        inp = t_client.InferInput('PYTHON_INPUT_0', input_data.shape, 'UINT8')
        inp.set_data_from_numpy(input_data)

        outp = t_client.InferRequestedOutput('PYTHON_OUTPUT_0')

        request_id = "req0"
        triton_client.async_stream_infer(model_name=model_name,
                                         inputs=[inp],
                                         request_id=request_id,
                                         outputs=[outp])

        recv_count = 0
        expected_count = 3
        result_dict = {}
        while recv_count < expected_count:
            data_item = user_data._completed_requests.get()
            if type(data_item) == InferenceServerException:
                raise data_item
            else:
                this_id = data_item.get_response().id
                if this_id not in result_dict.keys():
                    result_dict[this_id] = []
                result_dict[this_id].append(data_item)
            recv_count += 1

        print(result_dict)
        print(f'ITER {request_id}: OK')
