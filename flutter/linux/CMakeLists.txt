# Linux build for the ferro_ta FFI plugin.
# Bundles the prebuilt Rust .so into the consuming app's bundle lib directory.
cmake_minimum_required(VERSION 3.13)

set(PROJECT_NAME "ferro_ta")
project(${PROJECT_NAME} LANGUAGES CXX)

# ffiPlugin: no plugin class to compile. Declare the bundled library so the
# Flutter tool copies it into the app bundle's lib/ directory.
set(ferro_ta_bundled_libraries
  "${CMAKE_CURRENT_SOURCE_DIR}/lib/libferro_ta_flutter.so"
  PARENT_SCOPE
)
