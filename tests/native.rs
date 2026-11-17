use diffusion_cpp_rs::*;
use std::path::Path;

#[test]
fn tensor_round_trip_and_retained_lifetime() -> Result<()> {
    let tensor = Tensor::rgb(2, 1, &[1, 2, 3, 4, 5, 6])?;
    assert_eq!(tensor.shape(), [3, 2, 1]);
    assert_eq!(tensor.element_type(), TensorType::U8);
    let retained = tensor.clone();
    assert_eq!(retained.as_ptr(), tensor.as_ptr());
    drop(tensor);
    assert_eq!(retained.as_slice::<u8>()?, [1, 2, 3, 4, 5, 6]);
    assert_eq!(
        retained.as_slice::<f32>().unwrap_err().status,
        Status::TypeMismatch
    );
    Ok(())
}

#[test]
fn tensor_types_scalars_and_empty_dimensions() -> Result<()> {
    assert_eq!(Tensor::new(&[], &[42i32])?.as_slice::<i32>()?, [42]);
    assert_eq!(
        Tensor::new(&[2], &[i64::MIN, i64::MAX])?.as_slice::<i64>()?,
        [i64::MIN, i64::MAX]
    );
    assert_eq!(
        Tensor::new(&[2], &[1.25f32, -2.5])?.as_slice::<f32>()?,
        [1.25, -2.5]
    );
    assert_eq!(
        Tensor::new(&[2], &[1.25f64, -2.5])?.as_slice::<f64>()?,
        [1.25, -2.5]
    );
    let empty = Tensor::new::<u8>(&[3, 0, 8], &[])?;
    assert_eq!(empty.byte_len(), 0);
    assert_eq!(empty.shape(), [3, 0, 8]);
    assert!(empty.as_slice::<u8>()?.is_empty());
    Ok(())
}

#[test]
fn invalid_tensor_shapes_are_rejected() {
    for shape in [vec![-1], vec![1; 9], vec![3, 4], vec![]] {
        assert!(Tensor::new::<u8>(&shape, &[]).is_err());
    }
    assert_eq!(
        Tensor::new::<u8>(&[i64::MAX, i64::MAX], &[])
            .unwrap_err()
            .status,
        Status::OutOfRange
    );
    assert!(Tensor::rgb(0, 1, &[]).is_err());
}

#[test]
fn options_validate_type_purpose_and_range() -> Result<()> {
    let mut options = ImageOptions::new()?;
    options
        .set_string("prompt", "a cat")?
        .set_int("width", 512)?
        .set_float("strength", 0.5)?;
    assert_eq!(
        options
            .set_string("does_not_exist", "x")
            .unwrap_err()
            .status,
        Status::UnknownKey
    );
    assert_eq!(
        options.set_string("width", "512").unwrap_err().status,
        Status::TypeMismatch
    );
    assert_eq!(
        options.set_int("n_threads", 2).unwrap_err().status,
        Status::Unsupported
    );
    assert_eq!(
        options.set_int("width", 0).unwrap_err().status,
        Status::OutOfRange
    );
    assert!(options.set_int("width", i64::MAX).is_err());
    assert!(options.set_float("strength", f64::NAN).is_err());
    assert!(options.set_float("strength", 1.1).is_err());
    assert!(
        options
            .set_string("sample.method", "invalid-sampler")
            .is_err()
    );
    options.unset("width")?.unset("width")?;
    assert_eq!(
        options.unset("not_a_key").unwrap_err().status,
        Status::UnknownKey
    );
    Ok(())
}

#[test]
fn nul_bytes_and_native_diagnostics_are_owned() -> Result<()> {
    let mut options = ImageOptions::new()?;
    assert_eq!(
        options.set_string("prompt", "a\0b").unwrap_err().status,
        Status::InvalidArgument
    );
    assert_eq!(
        options.set_int("wid\0th", 512).unwrap_err().status,
        Status::InvalidArgument
    );
    let error = options.set_int("missing_key", 1).unwrap_err();
    let message = error.message.clone();
    assert!(!message.is_empty());
    options.set_int("width", 512)?;
    assert_eq!(error.message, message);
    Ok(())
}

#[test]
fn array_and_media_setters_retain_inputs() -> Result<()> {
    let mut image = ImageOptions::new()?;
    image
        .set_floats("sample.sigmas", &[1.0, 0.5, 0.0])?
        .set_ints("sample.guidance.slg.layers", &[1, 3])?
        .set_loras(
            "loras",
            &[Lora {
                path: Path::new("lora.safetensors"),
                multiplier: 0.75,
                is_high_noise: false,
            }],
        )?;
    assert!(image.set_floats("sample.sigmas", &[0.0, 1.0]).is_err());
    assert!(image.set_ints("sample.guidance.slg.layers", &[-1]).is_err());
    {
        let input = Tensor::rgb(2, 2, &[127; 12])?;
        image
            .set_tensor("init_image", &input)?
            .set_tensors("ref_images", &[input])?;
    }
    // Options still retain the input after all Rust-side tensor owners dropped.
    image.unset("init_image")?.unset("ref_images")?;
    let mut context = ContextOptions::new()?;
    context.set_embeddings(
        "embeddings",
        &[Embedding {
            name: "test",
            path: Path::new("embedding.pt"),
        }],
    )?;
    let audio = Audio {
        sample_rate: 16_000,
        data: Tensor::new(&[1, 4], &[0.0f32; 4])?,
    };
    let video = Video {
        fps: 24,
        data: Tensor::new(&[3, 2, 2, 1], &[0u8; 12])?,
        audio: Some(audio.clone()),
    };
    let mut options = VideoOptions::new()?;
    options
        .set_audios("ref_audios", &[audio])?
        .set_videos("ref_videos", &[video])?;
    options.unset("ref_audios")?.unset("ref_videos")?;
    Ok(())
}

#[test]
fn each_operation_has_a_distinct_options_type() -> Result<()> {
    UpscalerOptions::new()?.set_int("tile_size", 128)?;
    UpscaleOptions::new()?.set_tensor("image", &Tensor::rgb(2, 2, &[0; 12])?)?;
    VideoUpscalerOptions::new()?.set_int("upscale_factor", 2)?;
    VideoUpscaleOptions::new()?.set_tensor("frames", &Tensor::new(&[3, 2, 2, 1], &[0u8; 12])?)?;
    ADetailerOptions::new()?.set_int("n_threads", 1)?;
    ADetailOptions::new()?.set_string("adetail.prompt", "face")?;
    Ok(())
}

#[test]
fn missing_model_is_a_recoverable_error() -> Result<()> {
    let mut options = ContextOptions::new()?;
    options.set_path(
        "model_path",
        "/__diffusion_cpp_rs_missing__/model.safetensors",
    )?;
    let error = match Context::new(&options) {
        Ok(_) => panic!("a nonexistent model must not load"),
        Err(error) => error,
    };
    assert!(matches!(
        error.status,
        Status::LoadFailed | Status::InvalidArgument
    ));
    assert!(!error.message.is_empty());
    Ok(())
}

#[test]
fn canny_does_not_modify_input() -> Result<()> {
    let original = vec![127u8; 3 * 16 * 16];
    let image = Tensor::rgb(16, 16, &original)?;
    let edges = CannyOptions::default().process(&image)?;
    assert_eq!(image.as_slice::<u8>()?, original);
    assert_eq!(edges.shape(), image.shape());
    assert_eq!(edges.byte_len(), original.len());
    assert_ne!(edges.as_ptr(), image.as_ptr());
    Ok(())
}

#[test]
fn canny_rejects_invalid_numeric_inputs() -> Result<()> {
    let image = Tensor::rgb(2, 2, &[0; 12])?;
    for value in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
        let options = CannyOptions {
            strong: value,
            ..CannyOptions::default()
        };
        assert_eq!(
            options.process(&image).unwrap_err().status,
            Status::OutOfRange
        );
    }
    Ok(())
}

#[test]
fn native_build_is_linked() {
    assert!(!build_version().is_empty());
    assert!(!build_commit().is_empty());
    assert!(physical_cores() > 0);
}

/// Run explicitly with SD_TEST_MODEL=/path/to/sd1.5.safetensors and
/// `cargo test --test native model_inference -- --ignored --test-threads=1`.
#[test]
#[ignore = "requires a user-provided Stable Diffusion 1.x checkpoint"]
fn model_inference() -> Result<()> {
    let model =
        std::env::var_os("SD_TEST_MODEL").expect("set SD_TEST_MODEL to a local SD 1.x checkpoint");
    let mut loading = ContextOptions::new()?;
    loading.set_path("model_path", model)?;
    let mut context = Context::new(&loading)?;
    drop(loading);
    let mut options = ImageOptions::new()?;
    options
        .set_string("prompt", "a red apple")?
        .set_int("seed", 42)?
        .set_int("width", 64)?
        .set_int("height", 64)?
        .set_int("sample.steps", 1)?;
    let result = context.generate_image(&options)?;
    assert_eq!(result.kind(), ResultKind::Image);
    assert_eq!(result.image_count(), 1);
    let pixels = result.image(0)?;
    assert!(result.image(1).is_err());
    assert!(!result.generation_info(0)?.is_empty());
    assert!(result.audio()?.is_none());
    drop(result);
    drop(options);
    drop(context);
    assert_eq!(pixels.shape(), [3, 64, 64]);
    assert_eq!(pixels.as_slice::<u8>()?.len(), 3 * 64 * 64);
    Ok(())
}
