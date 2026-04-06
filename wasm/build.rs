fn main() {
    #[cfg(feature = "pffft")]
    build_pffft();
}

#[cfg(feature = "pffft")]
fn build_pffft() {
    println!("cargo:rerun-if-changed=vendor/pffft/pffft.c");
    println!("cargo:rerun-if-changed=vendor/pffft/pffft.h");

    cc::Build::new()
        .file("vendor/pffft/pffft.c")
        .include("vendor/pffft")
        .flag_if_supported("-O3")
        .compile("pffft");
}
