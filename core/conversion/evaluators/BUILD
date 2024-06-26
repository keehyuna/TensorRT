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
    name = "evaluators",
    srcs = [
        "NodeEvaluatorRegistry.cpp",
        "aten.cpp",
        "eval_macros.h",
        "eval_util.cpp",
        "eval_util.h",
        "prim.cpp",
    ],
    hdrs = [
        "evaluators.h",
    ],
    deps = [
        "//core/conversion/tensorcontainer",
        "//core/conversion/var",
        "//core/util:prelude",
    ] + select({
        ":windows": ["@libtorch_win//:libtorch"],
        ":use_pre_cxx11_abi": ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@libtorch"],
    }),
    alwayslink = True,
)

pkg_tar(
    name = "include",
    srcs = ["evaluators.h"],
    package_dir = "core/conversion/evaluators/",
)
