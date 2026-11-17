# diffusion-cpp-rs

[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) 的 Rust bindings，包含自动生成的完整 C FFI 和安全 Rust 封装。

固定上游版本：[`532d2c15c430a785a712b2d94281f04e733008a4`](https://github.com/leejet/stable-diffusion.cpp/tree/532d2c15c430a785a712b2d94281f04e733008a4)。使用这一版本的 opaque options / tensor / result API，与旧版 `new_sd_ctx`、`txt2img` 接口不兼容。

## 构建

需要 Rust 1.85+、C/C++17 编译器、CMake 3.20+、libclang 和 Git。macOS 需要 Xcode Command Line Tools；Linux 可安装 `build-essential cmake libclang-dev`。`bindgen` 找不到 libclang 时设置 `LIBCLANG_PATH`。

```sh
git clone https://github.com/jiabochao/diffusion-cpp-rs.git
cd diffusion-cpp-rs
git submodule update --init --depth 1 diffusion-cpp-sys/vendor/stable-diffusion.cpp
git -C diffusion-cpp-sys/vendor/stable-diffusion.cpp submodule update --init --depth 1 ggml
cargo build
cargo run --example system_info
```

只需初始化主源码和 `ggml`；库构建不需要 Web UI、libwebp、libwebm 子模块。Cargo 构建脚本不执行联网下载。首次构建会编译 C++，耗时明显长于增量构建。库静态链接进 Rust 程序。

| Feature | 用途 |
| --- | --- |
| 默认（无 feature） | CPU；macOS 使用 Accelerate |
| `metal` | Apple Metal；需要 Apple SDK |
| `cuda` | NVIDIA CUDA；需要 CUDA Toolkit 和对应编译工具链 |
| `vulkan` | Vulkan；需要 Vulkan SDK 和 shader 编译器 |
| `native` | 针对构建机器的 CPU 指令集优化；产物可能不能移到较旧的 CPU |

例如 `cargo build --features metal`。不要用 `--all-features` 代替平台选择。默认不编译可选的 CUTLASS 扩展内核；需要它时，初始化上游 `thirdparty/cutlass` 子模块，并通过 `CMAKE_ARGS='-DSD_CUDA_CUTLASS_PATH=/absolute/path/to/cutlass'` 指定目录。

## 快速使用

仓库尚未发布到 crates.io，可使用本地路径或 Git dependency：

```toml
[dependencies]
diffusion-cpp-rs = { git = "https://github.com/jiabochao/diffusion-cpp-rs.git" }
# 本地开发：diffusion-cpp-rs = { path = "../diffusion-cpp-rs", features = ["metal"] }
```

```rust,no_run
use diffusion_cpp_rs::{Context, ContextOptions, ImageOptions};

fn main() -> diffusion_cpp_rs::Result<()> {
    let mut loading = ContextOptions::new()?;
    loading.set_path("model_path", "model.safetensors")?
        .set_int("n_threads", 8)?;
    let mut context = Context::new(&loading)?;

    let mut options = ImageOptions::new()?;
    options.set_string("prompt", "a cat sitting in a sunny garden")?
        .set_string("negative_prompt", "blurry")?
        .set_int("width", 512)?
        .set_int("height", 512)?
        .set_int("seed", 42)?
        .set_int("sample.steps", 20)?
        .set_string("sample.method", "euler")?;

    let output = context.generate_image(&options)?;
    let image = output.image(0)?;
    println!("shape: {:?}", image.shape()); // [3, 512, 512]
    let rgb: &[u8] = image.as_slice()?;
    println!("{} bytes", rgb.len());
    Ok(())
}
```

提供本地 SD 1.x checkpoint 后，可以直接生成 PNG：

```sh
cargo run --release --example txt2img -- \
  /path/to/model.safetensors "a cat sitting in a sunny garden" output.png
# Apple GPU：在 --example 前加 --features metal
```

库和示例不会下载模型。分体模型（如 FLUX）可通过 `diffusion_model_path`、`clip_l_path`、`t5xxl_path`、`vae_path` 等 context 参数加载；具体组件与采样配置取决于模型。

## 接口范围

- `Context`：图片/视频生成、模型能力查询、ControlNet 加载/卸载、context 级进度回调。
- `Tensor`：U8/I32/I64/F32/F64 数据、形状和长度校验、只读切片、引用计数。
- `GenerationOutput`：图片批次、视频帧、连续视频张量、音频和 JSON 生成信息。
- `Upscaler`、`VideoUpscaler`、`ADetailer`：加载与执行接口。
- `CannyOptions`：图像边缘预处理。
- `sys` / `diffusion-cpp-sys`：完整原始 C API，包括模型转换、量化类型、取消、全局日志/预览回调、RPC 和 importance matrix。

每种操作有独立的 options 类型。所有 `set_*` 返回 `Result<&mut Self>`，可链式调用；未知键、类型错误、参数范围错误由上游校验并返回拥有独立错误消息的 `Error`。未设置的参数保留上游默认值，`unset` 恢复默认状态。

完整参数表见固定版本的 [C API 文档](https://github.com/leejet/stable-diffusion.cpp/blob/532d2c15c430a785a712b2d94281f04e733008a4/docs/guides/c_api.md)。例子：

```rust,ignore
// 图生图 / inpainting：像素为 RGB 交错排列。
let input = Tensor::rgb(width, height, &rgb_pixels)?;
options.set_tensor("init_image", &input)?.set_float("strength", 0.65)?;
// 单通道 mask 使用 Tensor::new(&[1, width, height], &mask_pixels)。

// 视频生成使用独立的参数类型。
let mut video_options = VideoOptions::new()?;
video_options.set_string("prompt", "a running dog")?
    .set_int("fps", 24)?;
let video_output = context.generate_video(&video_options)?;
```

张量形状以最连续维度在前：图片 `[C,W,H]`、音频 `[C,N]`、视频 `[C,W,H,T]`。`set_tensor`、数组和媒体 setter 会保留张量；字符串和描述符会被复制。`image()`、`audio()`、`video()` 返回独立持有引用的对象，即使释放原结果或 context 仍可使用。`Tensor::clone()` 只增加引用计数。

所有原生资源通过 `Drop` 自动释放。句柄保守地设为 `!Send + !Sync`；在同一工作线程创建、使用和释放 context。模型操作使用 `&mut self` 防止同一对象重入；不承诺多个 context 的并发原生推理安全性。进度回调要求 `Send + Sync + 'static`，可在回调中向 channel 发送事件；panic 不会穿过 C ABI。直接调用 `sys` 时，调用方负责原生全局状态、线程同步、指针有效性与回调生命周期。

## 使用外部源码或动态库

- `STABLE_DIFFUSION_SOURCE_DIR`：使用另一个包含 `ggml` 的源码目录。
- `STABLE_DIFFUSION_INCLUDE_DIR`：指定 `stable-diffusion.h` 所在目录。
- `STABLE_DIFFUSION_LIB_DIR`：跳过 C++ 构建，链接该目录中的共享 `stable-diffusion` 库。必须和生成 bindings 的头文件版本一致。
- `CMAKE_ARGS`：额外的空格分隔 `-DKEY=VALUE` 定义，例如 `-DCMAKE_CUDA_ARCHITECTURES=89`；不接受含空格的值。

共享库模式由调用者负责系统加载器路径（Linux 的 `LD_LIBRARY_PATH`、macOS 的 `DYLD_LIBRARY_PATH`、Windows 的 `PATH`）及 GPU runtime 依赖。不同上游版本之间不保证 API/ABI 兼容。

## 验证

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test --workspace --features metal # 仅 Apple
```

默认测试使用真实 C 库，覆盖张量数据与生命周期、参数验证、缺失模型错误、Canny 和回调 panic 隔离，不需要模型权重。可选推理测试需要本地 SD 1.x checkpoint：

```sh
SD_TEST_MODEL=/path/to/sd1.5.safetensors \
  cargo test --test native model_inference -- --ignored --test-threads=1
```

## 许可证

Rust bindings 使用 MIT。子模块及其依赖保留各自许可证；stable-diffusion.cpp 的许可证见其源码目录中的 `LICENSE`。
