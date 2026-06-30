fn main() {
    println!("cargo:rerun-if-env-changed=ORT_CXX_STDLIB");
    // On Linux wheel builds we statically link the C++ runtime (libstdc++) so the
    // wheel doesn't depend on the host's libstdc++.so.6 version: built on a newer
    // toolchain (manylinux GCC 11+) it would otherwise reference symbols like
    // `std::__throw_bad_array_new_length` (GLIBCXX_3.4.29) that are missing on
    // older systems (GCC 9/10).
    //
    // The build sets `ORT_CXX_STDLIB=` (empty) so ort-sys skips its own *dynamic*
    // stdc++ link; we then link it statically here. When the var is not set (a
    // normal `cargo build`/dev build), we leave the default dynamic link so the
    // build still works without a static libstdc++.a in the toolchain.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let static_cxx = matches!(std::env::var("ORT_CXX_STDLIB").as_deref(), Ok(""));
    if target_os == "linux" && static_cxx {
        // Link libstdc++ statically at the *end* of the link line (as a link-arg,
        // which the linker places after all rlibs/archives) so it resolves the
        // C++ symbols referenced by the ONNX Runtime archive and pulls in all the
        // members it needs. `cargo:rustc-link-lib=static=stdc++` placed it too
        // early in the order, leaving the C++ runtime unresolved.
        println!("cargo:rustc-link-arg=-Wl,--push-state,-Bstatic");
        println!("cargo:rustc-link-arg=-lstdc++");
        println!("cargo:rustc-link-arg=-Wl,--pop-state");
    }
}
