# Python environment for LITE-Edge v2. To use this, install nix with your
# Linux/MacOS package manager (use WSL if you're on Windows), cd to the v2
# directory, then just run nix-shell. You will then be placed into a shell
# environment that has everything you need installed.

# If you need to add python packages in the future, make sure you do it
# here.

{ pkgs ? import (builtins.fetchTarball {
  # Here, we pin a certain version of nixpkgs so everyone in the future
  # has the exact same python packages installed in their shell
  url = "https://github.com/nixos/nixpkgs/archive/92b504d2ff54385af14a1be2af7aedeb075bfba2.tar.gz";
  }) {} }:

pkgs.mkShell {
  packages = [
     (pkgs.python312.withPackages(pypkgs: with pypkgs; [ # Python 3.12
      tensorflow # 2.19.0
      numpy # 2,3,1
      matplotlib # 3.10.3
      pandas # 2.2.3
      scipy # 1-16.0
     ])) 
    ];
}
