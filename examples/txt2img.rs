use diffusion_cpp_rs::{Context, ContextOptions, ImageOptions};
use std::{
    env,
    fs::File,
    io::{self, BufWriter},
    path::PathBuf,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os().skip(1);
    let model = args
        .next()
        .ok_or("usage: txt2img MODEL 'PROMPT' [OUTPUT.png]")?;
    let prompt = args
        .next()
        .ok_or("a prompt is required")?
        .into_string()
        .map_err(|_| "prompt must be valid Unicode")?;
    let output = PathBuf::from(args.next().unwrap_or_else(|| "output.png".into()));
    if args.next().is_some() {
        return Err("too many arguments".into());
    }

    let mut loading = ContextOptions::new()?;
    loading.set_path("model_path", model)?;
    let mut context = Context::new(&loading)?;
    context.set_progress_callback(|event| {
        eprintln!(
            "{}: {}/{} {}",
            event.stage, event.completed, event.total, event.unit
        );
    });
    let mut options = ImageOptions::new()?;
    options
        .set_string("prompt", &prompt)?
        .set_string("negative_prompt", "blurry, low quality")?
        .set_int("width", 512)?
        .set_int("height", 512)?
        .set_int("seed", 42)?
        .set_int("sample.steps", 20)?;
    let result = context.generate_image(&options)?;
    let image = result.image(0)?;
    let shape = image.shape();
    if shape.len() != 3 || shape[0] != 3 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "expected an RGB image").into());
    }
    let file = BufWriter::new(File::create(&output)?);
    let mut encoder = png::Encoder::new(file, u32::try_from(shape[1])?, u32::try_from(shape[2])?);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.add_itxt_chunk("parameters".into(), result.generation_info(0)?)?;
    let mut writer = encoder.write_header()?;
    writer.write_image_data(image.as_slice::<u8>()?)?;
    writer.finish()?;
    println!("Saved {}", output.display());
    Ok(())
}
