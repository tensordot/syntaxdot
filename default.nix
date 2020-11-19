{ pkgs ? import <nixpkgs> {}
, naersk

, config
, cudaSupport ? config.cudaSupport or false

, withTfRecord ? false
, withHdf5 ? false
}:

with pkgs;

let
  naersk-lib = callPackage naersk {};
in naersk-lib.buildPackage {
  src = ./.;

  name = "syntaxdot";
  version = "0.1.0";

  nativeBuildInputs = [ cmake pkg-config ];

  buildInputs = [ libtorch-bin openssl ];

  singleStep = withTfRecord;

  cargoBuildOptions = o: o ++ [
      "--no-default-features"
      "--manifest-path syntaxdot-cli/Cargo.toml"
  ] ++ lib.optionals withTfRecord [
    "--features=tfrecord"
  ] ++ lib.optionals withHdf5 [
    "--features=load-hdf5"
  ];

  HDF5_DIR = symlinkJoin {
    name = "hdf5-join";
    paths = [ hdf5.dev hdf5.out ];
  };

  # torch-sys CUDA detection is a bit flaky. It looks for
  # $LIBTORCH/lib/libtorch_cuda.so, which does not exist with 
  # split outputs.
  LIBTORCH = if cudaSupport then
    symlinkJoin {
      name = "libtorch-join";
      paths = [ libtorch-bin.dev libtorch-bin.out ];
    }
    else libtorch-bin.dev;
}
