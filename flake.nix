{
  description = "SyntaxDot sequence labeler";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    crate2nix = {
      url = "github:kolloch/crate2nix";
      flake = false;
    };
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, crate2nix, nixpkgs, utils }:
    utils.lib.eachSystem  [ "x86_64-linux" ] (system:
    let
      models = import ./models.nix { inherit (pkgsWithCuda) fetchurl; };
      pkgsWithoutCuda = import nixpkgs {
        inherit system;
        config = {
          allowUnfreePredicate = pkg: builtins.elem (pkgsWithoutCuda.lib.getName pkg) [
            "libtorch"
          ];
        };
      };
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
      syntaxdot = pkgs:
      let
        crateOverrides = pkgs.callPackage build-support/crate-overrides.nix {};
        crateTools = pkgs.callPackage "${crate2nix}/tools.nix" {};
        buildRustCrate = pkgs.buildRustCrate.override {
          defaultCrateOverrides = crateOverrides;
        };
        cargoNix = pkgs.callPackage (crateTools.generatedCargoNix {
          name = "syntaxdot";
          src = ./.;
        }) {
          inherit buildRustCrate;
        };
      in cargoNix.workspaceMembers.syntaxdot-cli.build;
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

      packages.syntaxdot = syntaxdot pkgsWithoutCuda;

      packages.syntaxdotWithCuda = syntaxdot pkgsWithCuda;
    });
}
