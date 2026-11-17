fn main() {
    println!(
        "stable-diffusion.cpp {} ({})",
        diffusion_cpp_rs::build_version(),
        diffusion_cpp_rs::build_commit()
    );
    println!("Physical cores: {}", diffusion_cpp_rs::physical_cores());
    println!("{}", diffusion_cpp_rs::system_info());
    println!("Devices:\n{}", diffusion_cpp_rs::devices());
}
