#
# ferro_ta macOS plugin — bundles the prebuilt universal Rust dylib.
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
  s.dependency 'FlutterMacOS'
  s.platform = :osx, '10.14'

  # Universal (arm64 + x86_64) dylib produced by flutter-publish.yml.
  s.vendored_libraries = 'lib/libferro_ta_flutter.dylib'
end
