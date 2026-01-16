#!/bin/bash

set -ex

cd `dirname $0`
DEST_DIR="./build"

mkdir -p "$DEST_DIR"

rustup show
rustup target add wasm32-unknown-unknown

WASM_BINDGEN_VERSION=`cargo metadata --format-version=1 | jq -r '.packages[] | select(.name == "wasm-bindgen") | .version'`
if [ -z "$WASM_BINDGEN_VERSION" ]; then
    echo "Error: Could not find wasm-bindgen version in Cargo.toml"
    exit 1
fi
cargo install wasm-bindgen-cli --version "=$WASM_BINDGEN_VERSION" --root "$DEST_DIR" --force
