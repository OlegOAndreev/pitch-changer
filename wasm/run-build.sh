#!/bin/bash

set -e

# wasm-bindgen must be installed first using install-wasm-bindgen-cli.sh script

# Latest wasm-pack release was too long ago and now a few of dependencies are vulnerable, which is annoying.
# Run wasm-bindgen and wasm-opt ourselves as outlined here: https://fourteenscrews.com/essays/look-ma-no-wasm-pack/

BUILD_PROFILE="release"
#BUILD_PROFILE="debug"

echo "Building with profile: $BUILD_PROFILE"

cd `dirname $0`
BUILD_DIR="./build"
WASM_TARGET="wasm32-unknown-unknown"

echo "Building WebAssembly module..."
cargo build --target $WASM_TARGET --profile $BUILD_PROFILE

WASM_INPUT="./target/$WASM_TARGET/$BUILD_PROFILE/wasm_main_module.wasm"
if [ ! -f "$WASM_INPUT" ]; then
    echo "Error: WebAssembly file not found at $WASM_INPUT"
    exit 1
fi

# Do not run wasm-bindgen (relatively fast) and wasm-opt (relatively slow) on null builds.
CURRENT_HASH=`sha256sum "$WASM_INPUT"`
HASH_FILE="$BUILD_DIR/wasm_main_module.hash"
if [ -f "$HASH_FILE" ]; then
    PREVIOUS_HASH=`cat "$HASH_FILE"`
    if [ "$CURRENT_HASH" = "$PREVIOUS_HASH" ]; then
        echo "WASM unchanged, skipping wasm-bindgen and wasm-opt"
        exit 0
    fi
fi

echo "Running wasm-bindgen..."
time "$BUILD_DIR/bin/wasm-bindgen" --target web --out-dir "$BUILD_DIR" "$WASM_INPUT"

# We disable FinalizationRegistry for performance: registering/unregistering every return object by wasm-bindgen is
# very slow on Firefox and moderately slow on Chrome. FinalizationRegistry is a not so great idea anyway, e.g. see
# https://blog.cloudflare.com/en-en/we-shipped-finalizationregistry-in-workers-why-you-should-never-use-it/
#
# It would've been nice if this could be configured via cli flags...
echo "Patching wasm_main_module.js to disable FinalizationRegistry..."
# Use sed compatible with both macOS and Linux
sed -i.bak "s/(typeof FinalizationRegistry === 'undefined')/(true)/g" "$BUILD_DIR/wasm_main_module.js" && rm -f "$BUILD_DIR/wasm_main_module.js.bak"

echo "Running wasm-opt for optimization..."
WASM_OUTPUT="$BUILD_DIR/wasm_main_module_bg.wasm"
time npx wasm-opt -O -g "$WASM_OUTPUT" -o "$WASM_OUTPUT.opt"
mv "$WASM_OUTPUT.opt" "$WASM_OUTPUT"

echo "$CURRENT_HASH" > "$HASH_FILE"
