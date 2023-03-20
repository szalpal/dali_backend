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

import nvidia.dali as dali
import nvidia.dali.fn as fn
from nvidia.dali.plugin.triton import autoserialize


@autoserialize
@dali.pipeline_def(batch_size=3, num_threads=3, device_id=0,
                   output_dtype=dali.types.UINT8, output_ndim=[4, 4, 4])
def pipeline():
    vid1 = fn.experimental.inputs.video(name="INPUT_0", sequence_length=5, device='mixed')
    vid2 = fn.experimental.inputs.video(name="INPUT_1", sequence_length=5, device='mixed')
    vid3 = fn.experimental.inputs.video(name="INPUT_2", sequence_length=5, device='mixed')

    vid1 = fn.resize(vid1, resize_x=320, resize_y=240, name="OUTPUT_0")
    vid2 = fn.resize(vid2, resize_x=320, resize_y=240, name="OUTPUT_1")
    vid3 = fn.resize(vid3, resize_x=320, resize_y=240, name="OUTPUT_2")

    return vid1
