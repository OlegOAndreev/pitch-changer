#!/bin/bash

set -e

# wasm-bindgen must be installed first using install-build-deps.sh script

# Latest wasm-pack release was too long ago and now a few of dependencies are vulnerable, which is annoying.
# Run wasm-bindgen and wasm-opt ourselves as outlined here: https://fourteenscrews.com/essays/look-ma-no-wasm-pack/

BUILD_PROFILE=${BUILD_PROFILE:-release}

echo "Building with profile: $BUILD_PROFILE"

cd `dirname $0`
TOOLS_DIR="./target/tools"
BUILD_DIR="./pkg"
WASM_TARGET="wasm32-unknown-unknown"
BINARY_NAME="wasm_main_module"

case $BUILD_PROFILE in
  "dev")
    OUT_DIR="target/wasm32-unknown-unknown/debug"
    ;;
  "release")
    OUT_DIR="target/wasm32-unknown-unknown/release"
    ;;
  *)
    echo "Error: Unknown build profile \"$BUILD_PROFILE\", expected \"dev\" or \"release\""
    exit 1
esac

echo "Building WebAssembly module..."
cargo build --target $WASM_TARGET --profile $BUILD_PROFILE

WASM_INPUT="$OUT_DIR/$BINARY_NAME.wasm"
if [ ! -f "$WASM_INPUT" ]; then
    echo "Error: WebAssembly file not found at $WASM_INPUT"
    exit 1
fi

# Do not run wasm-bindgen and wasm-opt on null builds.
CURRENT_HASH=`sha256sum "$WASM_INPUT"`
HASH_FILE="$BUILD_DIR/$BINARY_NAME.hash"
if [ -f "$HASH_FILE" ]; then
    PREVIOUS_HASH=`cat "$HASH_FILE"`
    if [ "$CURRENT_HASH" = "$PREVIOUS_HASH" ]; then
        echo "WASM unchanged, skipping wasm-bindgen and wasm-opt"
        exit 0
    fi
fi

echo "Running wasm-bindgen..."
time "$TOOLS_DIR/bin/wasm-bindgen" \
    --target web --keep-debug --split-debug-info \
    --out-dir "$BUILD_DIR" "$WASM_INPUT"

# We disable FinalizationRegistry for performance: registering/unregistering every return object by wasm-bindgen is
# very slow on Firefox and moderately slow on Chrome. FinalizationRegistry is a not so great idea anyway, e.g. see
# https://blog.cloudflare.com/en-en/we-shipped-finalizationregistry-in-workers-why-you-should-never-use-it/
#
# It would've been nice if this could be configured via cli flags...
echo "Patching $BINARY_NAME.js to disable FinalizationRegistry..."
# Use sed compatible with both macOS and Linux
sed -i.bak "s/(typeof FinalizationRegistry === 'undefined')/(true)/g" "$BUILD_DIR/$BINARY_NAME.js" && rm -f "$BUILD_DIR/$BINARY_NAME.js.bak"

if [ $BUILD_PROFILE == "release" ]; then
  echo "Running wasm-opt for optimization..."
  WASM_OPT=../node_modules/binaryen/bin/wasm-opt
  WASM_OUTPUT="$BUILD_DIR/${BINARY_NAME}_bg.wasm"
  time $WASM_OPT -Os "$WASM_OUTPUT" -o "$WASM_OUTPUT"
else
  echo "Skipping wasm-opt"
fi

echo "$CURRENT_HASH" > "$HASH_FILE"
