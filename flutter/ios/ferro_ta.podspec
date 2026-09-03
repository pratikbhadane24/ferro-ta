#
# ferro_ta iOS plugin — bundles the prebuilt Rust static library as an
# xcframework. The `-force_load` flag ensures dart:ffi can resolve the
# flutter_rust_bridge symbols (static libs are otherwise dead-stripped).
#
Pod::Spec.new do |s|
  s.name             = 'ferro_ta'
  s.version          = '1.2.0'
  s.summary          = 'Fast technical-analysis indicators for Flutter (Rust FFI).'
  s.description      = <<-DESC
Rust-powered technical-analysis indicators for Flutter via flutter_rust_bridge.
                       DESC
  s.homepage         = 'https://github.com/pratikbhadane/ferro-ta'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'ferro-ta contributors' => 'pratikbhadane24@gmail.com' }
  s.source           = { :path => '.' }
  s.dependency 'Flutter'
  s.platform = :ios, '12.0'

  s.vendored_frameworks = 'ferro_ta_flutter.xcframework'

  # `-force_load` is required: the bridge symbols are only reached through
  # dart:ffi at runtime, so without it the linker dead-strips them from the
  # static library and lookups fail.
  #
  # The flag must be SDK-conditional. A single hardcoded slice forces the device
  # library into simulator builds too, which fails at link time (wrong
  # architecture). Slice directory names below match the xcframework produced by
  # flutter-publish.yml:
  #
  #   ferro_ta_flutter.xcframework/ios-arm64/libferro_ta_flutter.a
  #   ferro_ta_flutter.xcframework/ios-arm64_x86_64-simulator/libferro_ta_flutter.a
  #
  # Keep these in sync with the `xcodebuild -create-xcframework` step.
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'OTHER_LDFLAGS[sdk=iphoneos*]' =>
      '-Wl,-force_load,"${PODS_TARGET_SRCROOT}/ferro_ta_flutter.xcframework/ios-arm64/libferro_ta_flutter.a"',
    'OTHER_LDFLAGS[sdk=iphonesimulator*]' =>
      '-Wl,-force_load,"${PODS_TARGET_SRCROOT}/ferro_ta_flutter.xcframework/ios-arm64_x86_64-simulator/libferro_ta_flutter.a"',
  }
end
