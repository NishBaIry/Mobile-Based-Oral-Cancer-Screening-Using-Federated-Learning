#!/bin/bash

echo "🚀 Building Optimized Release APK"
echo "=================================="

cd "$(dirname "$0")"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
flutter clean

# Get dependencies
echo "📦 Getting dependencies..."
flutter pub get

# Build release APK
echo "🔨 Building release APK..."
flutter build apk --release --split-per-abi

echo ""
echo "✅ Build Complete!"
echo ""
echo "📦 Release APKs generated:"
ls -lh build/app/outputs/flutter-apk/app-*-release.apk

echo ""
echo "📱 APK Sizes:"
du -h build/app/outputs/flutter-apk/app-*-release.apk

echo ""
echo "💡 To install on your phone:"
echo "   flutter install --release"
echo ""
echo "📲 Or manually install:"
echo "   adb install build/app/outputs/flutter-apk/app-arm64-v8a-release.apk"
