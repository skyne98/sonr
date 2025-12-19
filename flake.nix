{
  description = "sonr - high-performance semantic search tool for local codebases";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
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
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default;

        rustPlatform = pkgs.makeRustPlatform {
          cargo = rustToolchain;
          rustc = rustToolchain;
        };

        commonArgs = {
          src = ./.;
          # This requires Cargo.lock to be present in the repository
          cargoLock.lockFile = ./Cargo.lock;

          nativeBuildInputs = [ pkgs.pkg-config ];
          buildInputs = [ pkgs.openssl ];

          # Tests require llama-server and network access for models
          doCheck = false;
        };

        sonr-daemon = rustPlatform.buildRustPackage (
          commonArgs
          // {
            pname = "sonr-daemon";
            version = "0.1.0";
            buildAndTestSubdir = "crates/sonr-daemon";
          }
        );

        sonr-cli = rustPlatform.buildRustPackage (
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
            postBuild = ''
              # Create a symlink for convenience if users expect 'sonr' command
              ln -s ${sonr-cli}/bin/sonr-cli $out/bin/sonr
            '';
          };
        };

        apps = {
          sonr-daemon = flake-utils.lib.mkApp { drv = sonr-daemon; };
          sonr-cli = flake-utils.lib.mkApp { drv = sonr-cli; };
          default = self.apps.${system}.sonr-cli;
        };

        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [
            rustToolchain
            pkgs.pkg-config
          ];
          buildInputs = [
            pkgs.openssl
            pkgs.llama-cpp
          ];
          shellHook = ''
            echo "Welcome to the sonr development environment"
            echo "llama-server: $(llama-server --version 2>/dev/null || echo 'not found in PATH')"
          '';
        };
      }
    );
}
