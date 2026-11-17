use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn enabled(feature: &str) -> bool {
    env::var_os(format!("CARGO_FEATURE_{feature}")).is_some()
}

fn on_off(value: bool) -> &'static str {
    if value { "ON" } else { "OFF" }
}

fn main() {
    for variable in [
        "STABLE_DIFFUSION_SOURCE_DIR",
        "STABLE_DIFFUSION_LIB_DIR",
        "STABLE_DIFFUSION_INCLUDE_DIR",
        "LIBCLANG_PATH",
        "CMAKE_ARGS",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=CMakeLists.txt");

    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let source = env::var_os("STABLE_DIFFUSION_SOURCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("vendor/stable-diffusion.cpp"));
    let include = env::var_os("STABLE_DIFFUSION_INCLUDE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| source.join("include"));
    let header = include.join("stable-diffusion.h");
    assert!(
        header.is_file(),
        "stable-diffusion.h not found at {}. Initialize the pinned source with `git submodule update --init --depth 1 diffusion-cpp-sys/vendor/stable-diffusion.cpp`, then initialize its ggml submodule. See README.md.",
        header.display()
    );

    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    bindgen::Builder::default()
        .header(header.to_string_lossy())
        .allowlist_function("(sd_.*|convert|convert_with_components|str_to_sd_type|.*imatrix.*)")
        .allowlist_type("(sd_.*|ggml_tensor)")
        .allowlist_var("SD_.*")
        .prepend_enum_name(false)
        .derive_default(false)
        .generate_comments(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("could not generate stable-diffusion.cpp bindings; install libclang")
        .write_to_file(out.join("bindings.rs"))
        .expect("could not write generated bindings");

    if let Some(directory) = env::var_os("STABLE_DIFFUSION_LIB_DIR") {
        // External mode expects a shared library with this exact C API. Its own
        // dependencies and runtime search path are the caller's responsibility.
        let directory = PathBuf::from(directory)
            .canonicalize()
            .expect("invalid STABLE_DIFFUSION_LIB_DIR");
        println!("cargo:rustc-link-search=native={}", directory.display());
        println!("cargo:rustc-link-lib=dylib=stable-diffusion");
        return;
    }

    assert!(
        source.join("ggml/CMakeLists.txt").is_file(),
        "ggml sources missing. Run `git -C diffusion-cpp-sys/vendor/stable-diffusion.cpp submodule update --init --depth 1 ggml`."
    );
    let target = env::var("TARGET").unwrap();
    assert!(
        !enabled("METAL") || target.contains("apple"),
        "the metal feature requires an Apple target"
    );
    // Watching native inputs also makes edits in an override checkout rebuild.
    for relative in [
        "CMakeLists.txt",
        "cmake",
        "include",
        "src",
        "thirdparty",
        "ggml/CMakeLists.txt",
        "ggml/cmake",
        "ggml/include",
        "ggml/src",
    ] {
        println!("cargo:rerun-if-changed={}", source.join(relative).display());
    }

    let mut config = cmake::Config::new(&root);
    config
        .define("SD_CPP_SOURCE_DIR", source.canonicalize().unwrap())
        .define("SD_CUDA", on_off(enabled("CUDA")))
        .define("SD_METAL", on_off(enabled("METAL")))
        .define("SD_VULKAN", on_off(enabled("VULKAN")))
        .define("GGML_NATIVE", on_off(enabled("NATIVE")))
        .profile("Release");
    if let Ok(arguments) = env::var("CMAKE_ARGS") {
        // Accept simple -DKEY=VALUE overrides without invoking a shell.
        for argument in arguments.split_whitespace() {
            let (key, value) = argument
                .strip_prefix("-D")
                .and_then(|arg| arg.split_once('='))
                .expect("CMAKE_ARGS accepts space-separated -DKEY=VALUE definitions");
            config.define(key, value);
        }
    }
    let destination = config.build();
    let libraries = fs::read_to_string(destination.join("native-libraries.txt"))
        .expect("CMake did not produce the native link manifest");
    for entry in libraries.lines().filter(|line| !line.is_empty()) {
        let (kind, path) = entry.split_once('|').expect("invalid native link manifest");
        link_library(kind, Path::new(path));
    }
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=dylib=c++");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        if enabled("METAL") {
            for framework in ["Foundation", "Metal", "MetalKit"] {
                println!("cargo:rustc-link-lib=framework={framework}");
            }
        }
    } else if !target.contains("msvc") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        if !target.contains("windows") {
            for library in ["m", "pthread", "dl"] {
                println!("cargo:rustc-link-lib={library}");
            }
        }
    }
}

fn link_library(kind: &str, path: &Path) {
    assert!(path.is_file(), "native library missing: {}", path.display());
    let file = path
        .file_name()
        .unwrap()
        .to_str()
        .expect("non-UTF-8 native library name");
    let name = file.strip_suffix(".lib").unwrap_or_else(|| {
        let name = file.strip_prefix("lib").unwrap_or(file);
        name.strip_suffix(".a")
            .or_else(|| name.strip_suffix(".dylib"))
            .or_else(|| name.split_once(".so").map(|(name, _)| name))
            .unwrap_or_else(|| panic!("unsupported native library name {file}"))
    });
    println!(
        "cargo:rustc-link-search=native={}",
        path.parent().unwrap().display()
    );
    println!("cargo:rustc-link-lib={kind}={name}");
}
