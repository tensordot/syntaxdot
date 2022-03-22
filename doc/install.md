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

## Nix/NixOS

### Nix with flake support

The SyntaxDot repository is a Nix flake. You can open a shell with SyntaxDot
as follows:

```shell
# SyntaxDot without CUDA support:
$ nix shell github:tensordot/syntaxdot

# SyntaxDot with CUDA support:
$ nix shell github:tensordot/syntaxdot#syntaxdotWithCuda 

# SyntaxDot is then available in the shell:
$ syntaxdot --version 
syntaxdot 0.3.0
```

You can also install a SyntaxDot package in your user profile:

```shell
# SyntaxDot without CUDA support:
$ nix profile install github:tensordot/syntaxdot

# SyntaxDot with CUDA support:
$ nix profile install github:tensordot/syntaxdot#syntaxdotWithCuda
```

### Nix without flake support

You can install SyntaxDot into your profile as follows:

```shell

# SyntaxDot without CUDA support:
$ nix-env \
  -f https://github.com/tensordot/syntaxdot/archive/main.tar.gz \
  -iA packages.x86_64-linux.syntaxdot

# SyntaxDot with CUDA support:
$ nix-env \
  -f https://github.com/tensordot/syntaxdot/archive/main.tar.gz \
  -iA packages.x86_64-linux.syntaxdotWithCuda
```

## Linux (other)

### Dependencies

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
$ export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${LIBTORCH}/lib
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
