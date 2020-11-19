{
  description = "SyntaxDot sequence labeler";

  inputs = {
    naersk = {
      url = "github:nmattia/naersk/master";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, naersk, nixpkgs, utils }:
    utils.lib.eachSystem  [ "x86_64-linux" ] (system:
    let
      pkgsWithCuda = import nixpkgs {
        inherit system;
        config = {
          allowUnfreePredicate = pkg: builtins.elem (pkgsWithCuda.lib.getName pkg) [
            "libtorch"
            "nvidia-x11"
          ];
          cudaSupport = true;
        };
      };
      pkgsWithoutCuda = import nixpkgs {
        inherit system;
        config = {
          allowUnfreePredicate = pkg: builtins.elem (pkgsWithCuda.lib.getName pkg) [
            "libtorch"
          ];
        };
      };
    in {
      devShell = pkgsWithCuda.callPackage ./shell.nix {};

      defaultPackage = self.packages.${system}.syntaxdot;

      packages = {
        syntaxdot = pkgsWithoutCuda.callPackage ./default.nix {
          inherit naersk;
        };

        syntaxdotWithCuda = pkgsWithCuda.callPackage ./default.nix {
          inherit naersk;
        };

        syntaxdotWithCudaFull = pkgsWithCuda.callPackage ./default.nix {
          inherit naersk;
          withHdf5 = true;
          withTfRecord = true;
        };
      };
    });
}
