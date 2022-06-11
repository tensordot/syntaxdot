fn main() {
    if tch::Cuda::is_available() {
        println!("cargo:rustc-cfg=test_cuda")
    }
}
