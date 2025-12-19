{
  description = "sonr - semantic search tool for local codebases";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      rust-overlay,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true;
        };

        rust = pkgs.rust-bin.stable.latest.default;

        commonArgs = {
          src = ./.;
          cargoLock.lockFile = ./Cargo.lock;
          nativeBuildInputs = [ pkgs.pkg-config ];
          buildInputs = [ pkgs.openssl ];
          doCheck = false; # Tests require llama-server and network access
        };

        sonr-daemon = pkgs.rustPlatform.buildRustPackage (
          commonArgs
          // {
            pname = "sonr-daemon";
            version = "0.1.0";
            buildAndTestSubdir = "crates/sonr-daemon";
          }
        );

        sonr-cli = pkgs.rustPlatform.buildRustPackage (
          commonArgs
          // {
            pname = "sonr-cli";
            version = "0.1.0";
            buildAndTestSubdir = "crates/sonr-cli";
          }
        );

      in
      {
        packages = {
          inherit sonr-daemon sonr-cli;
          default = pkgs.symlinkJoin {
            name = "sonr";
            paths = [
              sonr-daemon
              sonr-cli
            ];
          };
        };

        apps = {
          sonr-daemon = flake-utils.lib.mkApp { drv = sonr-daemon; };
          sonr-cli = flake-utils.lib.mkApp { drv = sonr-cli; };
          default = self.apps.${system}.sonr-cli;
        };

        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [
            rust
            pkgs.pkg-config
          ];
          buildInputs = [
            pkgs.openssl
            pkgs.llama-cpp
          ];
          shellHook = ''
            echo "sonr development environment"
            echo "llama-server version: $(llama-server --version 2>/dev/null || echo 'not found')"
          '';
        };
      }
    );
}
