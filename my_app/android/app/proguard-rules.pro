# Flutter wrapper
-keep class io.flutter.app.** { *; }
-keep class io.flutter.plugin.**  { *; }
-keep class io.flutter.util.**  { *; }
-keep class io.flutter.view.**  { *; }
-keep class io.flutter.**  { *; }
-keep class io.flutter.plugins.**  { *; }
-keep class io.flutter.embedding.** { *; }

# Google Play Core (for split APKs)
-keep class com.google.android.play.core.** { *; }
-dontwarn com.google.android.play.core.**

# AutoValue
-keep class com.google.auto.value.** { *; }
-dontwarn com.google.auto.value.**

# TensorFlow Lite
-keep class org.tensorflow.lite.** { *; }
-keep interface org.tensorflow.lite.** { *; }
-keep class org.tensorflow.lite.gpu.** { *; }
-dontwarn org.tensorflow.lite.**
-keepclassmembers class * {
    @org.tensorflow.lite.annotations.UsedByReflection *;
}

# TensorFlow Lite Support
-keep class org.tensorflow.lite.support.** { *; }
-dontwarn org.tensorflow.lite.support.**

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep model classes
-keep class com.example.my_app.MainActivity { *; }

# Preserve line numbers for debugging
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile

# General Android
-keepattributes *Annotation*
-keepattributes Signature
-keepattributes Exceptions
