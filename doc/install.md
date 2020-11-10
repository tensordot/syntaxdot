# Installation

## Dependencies

SyntaxDot has the following base requirements:

* A [Rust Toolchain](https://rustup.rs/)
* A C++ compiler
* libtorch
* cmake
* OpenSSL
* pkg-config

Additionally, compiling a SyntaxDot with training functionality requires:

* CUDA
* HDF5

### Fedora

Most of the dependencies can be installed in Fedora using the following
command:

```shell
$ sudo dnf install -y cmake gcc-c++ hdf5-devel openssl-devel pkg-config
```

Follow the RPM Fusion [instructions for installing
CUDA](https://rpmfusion.org/Howto/NVIDIA#CUDA). Besides these dependencies,
you also need a Rust toolchain and libtorch.

### Rust toolchain

A Rust stable toolchain can be installed through [rustup](https://rustup.rs/):

```shell
$ rustup default stable
```
### libtorch

[Download libtorch](https://pytorch.org/get-started/locally/) with the CXX11
ABI. A CUDA build is necessary for training, otherwise the CPU build can be
used. After unpacking the libtorch archive, you should set the following
environment variables:

```shell
$ export LIBTORCH=/path/to/libtorch
$ LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${LIBTORCH}/lib
```

## Building SyntaxDot

You can build SyntaxDot with support for training enabled using:

```shell
$ cargo install --path-syntaxdot-cli
```

To build SyntaxDot without training features, use:

```shell
$ cargo install --no-default-features --path syntaxdot-cli
```

The SyntaxDot binary will then be available in: ```~/.cargo/bin/syntaxdot```