#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenTopKConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=20]()
        %2 : int = prim::Constant[value=-1]()
        %3 : bool = prim::Constant[value=1]()
        %4 : bool = prim::Constant[value=1]()
        %5 : Tensor, %6 : Tensor = aten::topk(%0, %1, %2, %3, %4)
        return (%5, %6))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto dim0 = 10, dim1 = 10, dim2 = 100;

  // Initialize zero tensor to be filled with random indices along the final dimension
  auto in = at::zeros({dim0, dim1, dim2}, {at::kCUDA});

  // For each final dimension, fill it with random scramble of unique integers in the range [0, dim0*dim1*dim2)
  for (auto i = 0; i < dim0; i++) {
    for (auto j = 0; j < dim1; j++) {
      auto random_index_permutation = at::randperm(dim0 * dim1 * dim2, c10::kInt, {}, at::kCUDA, {}).slice(0, 0, dim2);
      in.slice(0, i, i + 1).slice(1, j, j + 1) = random_index_permutation;
    }
  }

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0])));
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[1], trt_results[1].reshape_as(jit_results[1])));
}

TEST(Converters, ATen1DTopKConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=20]()
        %2 : int = prim::Constant[value=-1]()
        %3 : bool = prim::Constant[value=1]()
        %4 : bool = prim::Constant[value=1]()
        %5 : Tensor, %6 : Tensor = aten::topk(%0, %1, %2, %3, %4)
        return (%5, %6))IR";
  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::rand({100}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0]));
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[1], trt_results[1]));
}
