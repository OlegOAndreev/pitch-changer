#!/bin/bash

set -e

export CMAKE_BUILD_TYPE=Debug
# export CMAKE_BUILD_TYPE=Release
export CMAKE_EXPORT_COMPILE_COMMANDS=ON

BUILD_DIR_NAME=build

SOURCE_DIR=`dirname $0`
BUILD_DIR=$SOURCE_DIR/$BUILD_DIR_NAME
if [[ ! -d $BUILD_DIR || -n "$CMAKE_RECONFIGURE" ]]; then
    mkdir -p $BUILD_DIR
    time emcmake cmake -S $SOURCE_DIR -B $BUILD_DIR
fi
time cmake --build $BUILD_DIR -v
