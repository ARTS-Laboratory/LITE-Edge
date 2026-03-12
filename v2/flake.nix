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
              numpy # 2.3.1
              matplotlib # 3.10.3
              pandas # 2.2.3
              scipy # 1-16.0
              keras
              pip # Do not use this. This is just for creating Requirements.txt.
             ])) 
            ];

          # We would like a Requirements.txt to be generated when this flake
          # is built so conda/pip users can be kept in-the loop with updated
          # dependencies. Since the environment the flake built is minimal,
          # this Requirements.txt is minimal and cross-platform.
          shellHook = ''
            # Use git to see if this flake was edited
            if [[ -n $(git status --porcelain -- ./flake.nix) ]]; then
              pip freeze > Requirements.txt
              # Only remind the user if the change actually modified dependencies
              if [[ -n $(git status --porcelain -- ./Requirements.txt) ]]; then
                echo "Requirements.txt has been updated. Please remember to commit your changes."
              fi
            fi
          '';
        };
      }
    );
}
