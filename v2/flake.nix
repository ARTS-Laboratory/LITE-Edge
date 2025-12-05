{
  description = "A basic flake with a shell";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.systems.url = "github:nix-systems/default";
  inputs.flake-utils = {
    url = "github:numtide/flake-utils";
    inputs.systems.follows = "systems";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
             (pkgs.python312.withPackages(pypkgs: with pypkgs; [ # Python 3.12
              tensorflow # 2.19.0
              numpy # 2,3,1
              matplotlib # 3.10.3
              pandas # 2.2.3
              scipy # 1-16.0
              keras
             ])) 
            ];
        };
      }
    );
}
