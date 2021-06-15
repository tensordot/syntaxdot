{
  description = "SyntaxDot sequence labeler";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, utils }:
    utils.lib.eachSystem  [ "x86_64-linux" ] (system:
    let
      models = import ./models.nix { inherit (pkgsWithCuda) fetchurl; };
      allowLibTorch = pkgs: pkg: builtins.elem (pkgs.lib.getName pkg) [
        "libtorch"
      ];
      pkgsWithoutCuda = import nixpkgs {
        inherit system;
        config = {
          allowUnfreePredicate = allowLibTorch pkgsWithoutCuda;
        };
      };
      pkgsWithCuda = import nixpkgs {
        inherit system;
        config = {
          allowUnfreePredicate = allowLibTorch pkgsWithCuda;
          cudaSupport = true;
        };
      };
      version = (builtins.fromTOML (builtins.readFile syntaxdot-cli/Cargo.toml)).package.version;
      syntaxdot = pkgs: with pkgs; rustPlatform.buildRustPackage {
        inherit version;

        pname = "syntaxdot";

        src = builtins.path {
          name = "syntaxdot";
          path = ./.;
        };

        cargoLock = {
          lockFile = ./Cargo.lock;
        };

        nativeBuildInputs = [ cmake pkg-config ];

        buildInputs = [ openssl ];

        LIBTORCH = symlinkJoin {
          name = "torch-join";
          paths = [ libtorch-bin.dev libtorch-bin.out ];
        };

        meta = with lib; {
          description = "Multi-task syntax annotator using transformers";
          homepage = "https://github.com/tensordot/syntaxdot";
          license = licenses.blueOak100;
          platforms = [ "x86_64-linux" ];
          maintainers = with maintainers; [ danieldk ];
        };
      };
    in {
      defaultPackage = self.packages.${system}.syntaxdot;

      devShell = with pkgsWithCuda; mkShell (models // {
        nativeBuildInputs = [ cmake pkg-config rustup ];

        buildInputs = [ openssl ];

        LIBTORCH = symlinkJoin {
          name = "torch-join";
          paths = [ libtorch-bin.dev libtorch-bin.out ];
        };
      });

      packages = {
        syntaxdot = syntaxdot pkgsWithoutCuda;
        syntaxdotWithCuda = syntaxdot pkgsWithCuda;
      };
    });
}
