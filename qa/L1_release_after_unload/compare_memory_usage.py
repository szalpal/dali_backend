#!/usr/bin/env python

# The MIT License (MIT)
#
# Copyright (c) 2024 NVIDIA CORPORATION
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

import sys
import xml.etree.ElementTree


def get_memory_usage(nvidia_smi_xml_filename):
    """
    Returns the value of currently used FB memory in MiB,
    as reported by nvidia-smi.
    
    Parameters:
    -----------
        nvidia_smi_xml_filename: The filename of XML generated by a command, like:
                                 nvidia-smi -q -i 0 -x | tee gpu.xml
    """
    tree = xml.etree.ElementTree.parse(nvidia_smi_xml_filename)
    root = tree.getroot()
    fb_usage = root.find(".//gpu/fb_memory_usage/used").text
    values = fb_usage.split()
    assert values[1] == 'MiB', "The value from the nvidia-smi log is not in MiB." \
                               "You may need to implement the functionality."
    return int(values[0])


def test_condition(pre, post, epsilon):
    return pre + epsilon > post


def main():
    mem_usage_pre = get_memory_usage('/tmp/mu_pre.xml')
    mem_usage_post = get_memory_usage('/tmp/mu_post.xml')
    if not test_condition(mem_usage_pre, mem_usage_post, 50):
        sys.exit(1)


if __name__ == '__main__':
    main()
