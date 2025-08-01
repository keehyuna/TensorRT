load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_torch_whl",
    flag_values = {
        "//toolchains/dep_src:torch": "whl"
    },
)

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

config_setting(
    name = "jetpack",
    constraint_values = [
        "@platforms//cpu:aarch64",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "jetpack"
    },
)


test_suite(
    name = "api_tests",
    tests = [
        ":test_collections",
        ":test_compiled_modules",
        ":test_default_input_types",
        ":test_dynamic_fallback",
        ":test_dynamic_size",
        ":test_example_tensors",
        ":test_module_fallback",
        ":test_modules_as_engines",
        ":test_multiple_registered_engines",
        ":test_runtime_thread_safety",
        ":test_serialization",
    ],
)

test_suite(
    name = "aarch64_api_tests",
    tests = [
        ":test_collections",
        ":test_compiled_modules",
        ":test_default_input_types",
        ":test_dynamic_fallback",
        ":test_dynamic_size",
        ":test_example_tensors",
        ":test_module_fallback",
        ":test_modules_as_engines",
        ":test_multiple_registered_engines",
        ":test_runtime_thread_safety",
        ":test_serialization",
    ],
)

cc_test(
    name = "test_default_input_types",
    srcs = ["test_default_input_types.cpp"],
    data = [
        "//tests/modules:jit_models",
    ],
    deps = [
        ":cpp_api_test",
    ],
)

cc_test(
    name = "test_example_tensors",
    srcs = ["test_example_tensors.cpp"],
    data = [
        "//tests/modules:jit_models",
    ],
    deps = [
        ":cpp_api_test",
    ],
)

cc_test(
    name = "test_serialization",
    srcs = ["test_serialization.cpp"],
    data = [
        "//tests/modules:jit_models",
    ],
    deps = [
        ":cpp_api_test",
    ],
)

cc_test(
    name = "test_multiple_registered_engines",
    srcs = ["test_multiple_registered_engines.cpp"],
    data = [
        "//tests/modules:jit_models",
    ],
    deps = [
        "//tests/util",
        "@googletest//:gtest_main",
    ] + select({
        ":windows": ["@libtorch_win//:libtorch"],
        ":use_torch_whl": ["@torch_whl//:libtorch"],
        ":jetpack": ["@torch_l4t//:libtorch"],
        "//conditions:default": ["@libtorch"],
    }),
)

cc_test(
    name = "test_modules_as_engines",
    timeout = "long",
    srcs = ["test_modules_as_engines.cpp"],
    data = [
        "//tests/modules:jit_models",
    ],
    deps = [
        ":cpp_api_test",
    ],
)

cc_test(
    name = "test_runtime_thread_safety",
    srcs = ["test_runtime_thread_safety.cpp"],
    data = [
        "//tests/modules:jit_models",
    ],
    deps = [
        ":cpp_api_test",
    ],
)

cc_test(
    name = "test_module_fallback",
    srcs = ["test_module_fallback.cpp"],
    data = [
        "//tests/modules:jit_models",
    ],
    deps = [
        "//tests/util",
        "@googletest//:gtest_main",
    ] + select({
        ":windows": ["@libtorch_win//:libtorch"],
        ":use_torch_whl": ["@torch_whl//:libtorch"],
        ":jetpack": ["@torch_l4t//:libtorch"],
        "//conditions:default": ["@libtorch"],
    }),
)

cc_test(
    name = "test_dynamic_fallback",
    srcs = ["test_dynamic_fallback.cpp"],
    data = [
        "//tests/modules:jit_models",
    ],
    deps = [
        "//tests/util",
        "@googletest//:gtest_main",
    ] + select({
        ":windows": ["@libtorch_win//:libtorch"],
        ":use_torch_whl": ["@torch_whl//:libtorch"],
        ":jetpack": ["@torch_l4t//:libtorch"],
        "//conditions:default": ["@libtorch"],
    }),
)

cc_test(
    name = "test_dynamic_size",
    srcs = ["test_dynamic_size.cpp"],
    deps = [
        "//tests/util",
        "@googletest//:gtest_main",
    ] + select({
        ":windows": ["@libtorch_win//:libtorch"],
        ":use_torch_whl": ["@torch_whl//:libtorch"],
        ":jetpack": ["@torch_l4t//:libtorch"],
        "//conditions:default": ["@libtorch"],
    }),
)

cc_test(
    name = "test_collections",
    srcs = ["test_collections.cpp"],
    data = [
        "//tests/modules:jit_models",
    ],
    deps = [
        "//tests/util",
        "@googletest//:gtest_main",
    ] + select({
        ":windows": ["@libtorch_win//:libtorch"],
        ":use_torch_whl": ["@torch_whl//:libtorch"],
        ":jetpack": ["@torch_l4t//:libtorch"],
        "//conditions:default": ["@libtorch"],
    }),
)

cc_test(
    name = "test_compiled_modules",
    timeout = "long",
    srcs = ["test_compiled_modules.cpp"],
    data = [
        "//tests/modules:jit_models",
    ],
    deps = [
        ":cpp_api_test",
    ],
)

cc_test(
    name = "test_multi_gpu_serde",
    srcs = ["test_multi_gpu_serde.cpp"],
    data = [
        "//tests/modules:jit_models",
    ],
    deps = [
        ":cpp_api_test",
    ],
)

cc_library(
    name = "cpp_api_test",
    hdrs = ["cpp_api_test.h"],
    deps = [
        "//tests/util",
        "@googletest//:gtest_main",
    ] + select({
        ":windows": ["@libtorch_win//:libtorch"],
        ":use_torch_whl": ["@torch_whl//:libtorch"],
        ":jetpack": ["@torch_l4t//:libtorch"],
        "//conditions:default": ["@libtorch"],
    }),
)
