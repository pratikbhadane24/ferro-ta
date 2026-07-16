# Windows build for the ferro_ta FFI plugin.
# Bundles the prebuilt Rust .dll into the consuming app's runtime directory.
cmake_minimum_required(VERSION 3.14)

set(PROJECT_NAME "ferro_ta")
project(${PROJECT_NAME} LANGUAGES CXX)

# ffiPlugin: no plugin class to compile. Just declare the bundled library so the
# Flutter tool copies it next to the app executable.
set(ferro_ta_bundled_libraries
  "${CMAKE_CURRENT_SOURCE_DIR}/lib/ferro_ta_flutter.dll"
  PARENT_SCOPE
)
