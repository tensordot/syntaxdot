{ pkgs ? import <nixpkgs> {} }:

with pkgs;

mkShell {
  nativeBuildInputs = [ cmake pkg-config rustup ];

  buildInputs = [ openssl ];

  HDF5_DIR = symlinkJoin {
    name = "hdf5-join";
    paths = [ hdf5.dev hdf5.out ];
  };

  LIBTORCH = symlinkJoin {
    name = "torch-join";
    paths = [ libtorch-bin.dev libtorch-bin.out ];
  };
}
