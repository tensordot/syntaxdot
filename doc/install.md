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

## Linux/macOS

### Dependencies

#### macOS

Install `cmake`, for instance through Homebrew:

```shell
brew install cmake
```

#### Fedora

Most of the dependencies can be installed in Fedora using the following
command:

```shell
$ sudo dnf install -y cmake gcc-c++ openssl-devel pkg-config
```

Follow the RPM Fusion [instructions for installing
CUDA](https://rpmfusion.org/Howto/NVIDIA#CUDA). Besides these dependencies,
you also need a Rust toolchain and libtorch (see below).

#### Debian/Ubuntu

Install the following dependencies through APT:

```shell
$ apt-get install -y build-essential cmake libssl-dev pkg-config
```

For installing CUDA, please refer to your distribution's instructions.
Besides these dependencies, you also need a Rust toolchain and libtorch
(see below).

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
# Linux:
$ export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${LIBTORCH}/lib
# macOS
$ export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH:+$DYLD_LIBRARY_PATH:}${LIBTORCH}/lib
```

There are currently no libtorch releases available for macOS ARM64. However,
you can use libtorch from the `torch` Python package:

```shell
$ pip install torch
$ export LIBTORCH=$(python -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')
$ export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH:+$DYLD_LIBRARY_PATH:}${LIBTORCH}/lib
```

## Building SyntaxDot

You can build SyntaxDot with support for training enabled using:

```shell
$ cargo install syntaxdot-cli
```

To build SyntaxDot without training features, use:

```shell
$ cargo install --no-default-features syntaxdot-cli
```

The SyntaxDot binary will then be available in: ```~/.cargo/bin/syntaxdot```
