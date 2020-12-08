{ pkgs ? import <nixpkgs> {} }:

with pkgs;

mkShell {
  nativeBuildInputs = [ cmake pkg-config rustup ];

  buildInputs = [ openssl ];

  LIBTORCH = symlinkJoin {
    name = "torch-join";
    paths = [ libtorch-bin.dev libtorch-bin.out ];
  };
}
