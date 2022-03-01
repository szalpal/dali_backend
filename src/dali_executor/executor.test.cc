// The MIT License (MIT)
//
// Copyright (c) 2020 NVIDIA CORPORATION
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <algorithm>
#include <catch2/catch.hpp>

#include "src/dali_executor/dali_executor.h"
#include "src/dali_executor/test/test_utils.h"
#include "src/dali_executor/test_data.h"
#include "src/model_provider/model_provider.h"

namespace triton { namespace backend { namespace dali { namespace test {

template<typename T, typename Op>
void coalesced_compare(const std::vector<OBufferDescr> &obuffers,
                       const std::vector<std::vector<T>> &ibuffers, size_t inp_size, const Op &op) {
  size_t inp_buff_i = 0;
  size_t inp_i = 0;
  size_t out_buff_i = 0;
  size_t out_i = 0;
  std::vector<T> obuffer;
  for (size_t i = 0; i < inp_size; ++i) {
    if (inp_i == ibuffers[inp_buff_i].size()) {
      inp_i = 0;
      inp_buff_i++;
    }
    if (out_i == obuffers[out_buff_i].size / sizeof(T)) {
      out_i = 0;
      out_buff_i++;
    }
    if (out_i == 0) {
      auto descr = obuffers[out_buff_i];
      REQUIRE(descr.size % sizeof(T) == 0);
      obuffer.resize(descr.size / sizeof(T));
      MemCopy(CPU, obuffer.data(), descr.device, descr.data, descr.size);
    }
    REQUIRE(obuffer[out_i] == op(ibuffers[inp_buff_i][inp_i]));
    out_i++;
    inp_i++;
  }
}

TEST_CASE("Identity Pipeline") {
  std::string serialized_pipeline_path="/home/mszolucha/clion_deploy/Triton/dali_backend/qa/L0_identity_cpu/model_repository/dali_identity_cpu/1/model.dali";
  FileModelProvider mp(serialized_pipeline_path);
  DaliPipeline pipeline(mp.GetModel(), 256, 4, ::dali::CPU_ONLY_DEVICE_ID);
  DaliExecutor executor(std::move(pipeline));
  std::mt19937 rand(1217);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  const std::string inp_name = "DALI_INPUT_0";
  auto scaling_test = [&](const std::vector<int> &batch_sizes,
                          const std::vector<int> &out_batch_sizes,
                          const std::vector<device_type_t> &out_devs) {
    REQUIRE(std::accumulate(batch_sizes.begin(), batch_sizes.end(), 0) ==
            std::accumulate(out_batch_sizes.begin(), out_batch_sizes.end(), 0));
    REQUIRE(out_devs.size() == out_batch_sizes.size());
    std::vector<TensorListShape<>> shapes;
    for (auto batch_size : batch_sizes) {
      TensorListShape<> shape(batch_size, 2);
      for (int i = 0; i < batch_size; ++i) {
        shape.set_tensor_shape(i, TensorShape<>(i + 1, 50));
      }
      shapes.push_back(shape);
    }
    std::vector<std::vector<float>> input_buffers;
    auto input = RandomInput(input_buffers, inp_name, shapes, [&]() { return dist(rand); });
    std::cout<<"DUPA1\n";
    auto output = executor.Run({input});
    std::cout<<"DUPA2\n";
    REQUIRE(cat_list_shapes(shapes) == output[0].shape);
    size_t inp_size = 0;
    for (auto &inp_buffer : input_buffers)
      inp_size += inp_buffer.size();
    std::vector<std::unique_ptr<IOBufferI>> output_buffers;
    int ti = 0;
    for (size_t out_i = 0; out_i < out_batch_sizes.size(); ++out_i) {
      int64_t buffer_vol = 0;
      for (int i = 0; i < out_batch_sizes[out_i]; ++i) {
        buffer_vol += volume(output[0].shape[ti]) * sizeof(float);
        ti++;
      }
      if (out_devs[out_i] == device_type_t::CPU) {
        output_buffers.emplace_back(std::make_unique<IOBuffer<CPU>>(buffer_vol));
      } else {
        output_buffers.emplace_back(std::make_unique<IOBuffer<GPU>>(buffer_vol));
      }
    }
    std::vector<ODescr> output_vec(1);
    auto &outdesc = output_vec[0];
    for (auto &out_buffer : output_buffers) {
      outdesc.buffers.push_back(out_buffer->get_descr());
    }
    executor.PutOutputs(output_vec);
    coalesced_compare(outdesc.buffers, input_buffers, inp_size, [](float a) { return a * 2; });
  };

  SECTION("Simple execute") {
    scaling_test({3, 2, 1}, {6}, {CPU});
  }

  SECTION("Chunked output") {
    scaling_test({3, 3}, {3, 3}, {CPU, CPU});
  }
}

}}}}  // namespace triton::backend::dali::test
