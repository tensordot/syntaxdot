{
  description = "SyntaxDot sequence labeler";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, utils }:
    utils.lib.eachSystem  [ "x86_64-linux" ] (system:
    let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfreePredicate = pkg: builtins.elem (pkgs.lib.getName pkg) [
            "libtorch"
            "nvidia-x11"
          ];
          cudaSupport = true;
        };
      };
    in {
      devShell = import ./shell.nix { inherit pkgs; };
    });
}
