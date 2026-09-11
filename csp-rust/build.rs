//! Builds the caracal7 C API (`cli/ffi.mojo`) as a shared library with the repo's Mojo toolchain and
//! links it. `CARACAL7_REPO` names the caracal7-prover checkout; the default is the parent directory.

use std::{env, path::PathBuf, process::Command};

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo = env::var("CARACAL7_REPO")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest.join(".."));
    assert!(
        repo.join("cli/ffi.mojo").exists(),
        "CARACAL7_REPO ({}) is not a caracal7-prover checkout",
        repo.display()
    );
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let lib = out.join("libcaracal7.dylib");
    let status = Command::new("uv")
        .args(["run", "mojo", "build", "--Werror", "--emit", "shared-lib", "-I", "src", "cli/ffi.mojo", "-o"])
        .arg(&lib)
        .current_dir(&repo)
        .status()
        .expect("uv run mojo build");
    assert!(status.success(), "mojo build of cli/ffi.mojo failed");
    // The Mojo runtime dylibs the library links against live in the repo's venv.
    let rt = Command::new("uv")
        .args(["run", "python", "-c", "import modular, os; print(os.path.join(os.path.dirname(modular.__file__), 'lib'))"])
        .current_dir(&repo)
        .output()
        .expect("uv run python");
    let rt = String::from_utf8(rt.stdout).unwrap().trim().to_string();
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=dylib=caracal7");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", out.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{rt}");
    println!("cargo:rerun-if-changed={}", repo.join("cli/ffi.mojo").display());
    println!("cargo:rerun-if-changed={}", repo.join("src").display());
    println!("cargo:rerun-if-env-changed=CARACAL7_REPO");
}
