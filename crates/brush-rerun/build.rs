fn main() {
    // rerun is only built for desktop. Keep in sync with the target-specific
    // dependencies in Cargo.toml.
    println!("cargo::rustc-check-cfg=cfg(rerun)");
    let family = std::env::var("CARGO_CFG_TARGET_FAMILY").unwrap_or_default();
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if std::env::var_os("CARGO_FEATURE_RERUN").is_some()
        && !family.split(',').any(|f| f == "wasm")
        && os != "ios"
        && os != "android"
    {
        println!("cargo::rustc-cfg=rerun");
    }
}
