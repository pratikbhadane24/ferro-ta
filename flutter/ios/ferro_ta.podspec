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

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'OTHER_LDFLAGS' => '-Wl,-force_load,"${PODS_TARGET_SRCROOT}/ferro_ta_flutter.xcframework/ios-arm64/libferro_ta_flutter.a"',
  }
end
