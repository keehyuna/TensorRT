load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_pkg//:pkg.bzl", "pkg_tar")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    },
)

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

cc_library(
    name = "weights",
    srcs = [
        "Weights.cpp",
    ],
    hdrs = [
        "Weights.h",
    ],
    deps = [
        "//core/conversion/conversionctx",
        "//core/util:prelude",
    ] + select({
        ":windows": ["@tensorrt_win//:nvinfer", "@libtorch_win//:libtorch"],
        ":use_pre_cxx11_abi": ["@tensorrt//:nvinfer", "@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@tensorrt//:nvinfer", "@libtorch"],
    }),
    alwayslink = True,
)

cc_library(
    name = "converter_util",
    srcs = [
        "converter_util.cpp",
    ],
    hdrs = [
        "converter_util.h",
    ],
    deps = [
        ":weights",
        "//core/conversion/conversionctx",
        "//core/util:prelude",
    ] + select({
        ":windows": ["@tensorrt_win//:nvinfer", "@libtorch_win//:libtorch"],
        ":use_pre_cxx11_abi": ["@tensorrt//:nvinfer", "@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@tensorrt//:nvinfer", "@libtorch"],
    }),
    alwayslink = True,
)

cc_library(
    name = "converters",
    srcs = [
        "NodeConverterRegistry.cpp",
        "impl/activation.cpp",
        "impl/batch_norm.cpp",
        "impl/bitwise.cpp",
        "impl/cast.cpp",
        "impl/chunk.cpp",
        "impl/concat.cpp",
        "impl/constant.cpp",
        "impl/constant_pad.cpp",
        "impl/conv_deconv.cpp",
        "impl/cumsum.cpp",
        "impl/einsum.cpp",
        "impl/element_wise.cpp",
        "impl/expand.cpp",
        "impl/internal_ops.cpp",
        "impl/interpolate.cpp",
        "impl/layer_norm.cpp",
        "impl/linear.cpp",
        "impl/lstm_cell.cpp",
        "impl/matrix_multiply.cpp",
        "impl/max.cpp",
        "impl/normalize.cpp",
        "impl/pooling.cpp",
        "impl/quantization.cpp",
        "impl/reduce.cpp",
        "impl/reflection_pad.cpp",
        "impl/replication_pad.cpp",
        "impl/select.cpp",
        "impl/shuffle.cpp",
        "impl/softmax.cpp",
        "impl/squeeze.cpp",
        "impl/stack.cpp",
        "impl/topk.cpp",
        "impl/unary.cpp",
        "impl/unsqueeze.cpp",
    ],
    hdrs = [
        "converters.h",
    ],
    deps = [
        ":converter_util",
        "//core/conversion/conversionctx",
        "//core/conversion/tensorcontainer",
        "//core/conversion/var",
        "//core/plugins:torch_tensorrt_plugins",
        "//core/util:prelude",
    ] + select({
        ":windows": ["@tensorrt_win//:nvinfer", "@libtorch_win//:libtorch"],
        ":use_pre_cxx11_abi": ["@tensorrt//:nvinfer", "@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@tensorrt//:nvinfer", "@libtorch"],
    }),
    alwayslink = True,
)

pkg_tar(
    name = "include",
    srcs = [
        "Weights.h",
        "converter_util.h",
        "converters.h",
    ],
    package_dir = "core/conversion/converters/",
)
