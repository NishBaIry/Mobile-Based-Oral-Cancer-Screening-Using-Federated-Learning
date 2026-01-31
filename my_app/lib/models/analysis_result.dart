import 'dart:convert';
import 'dart:typed_data';

class AnalysisResult {
  final String id;
  final DateTime timestamp;
  final String imageBase64; // Store annotated image as base64
  final bool yoloDetected;
  final String lesionType;
  final double yoloConfidence;
  final String mobileNetClassification;
  final double mobileNetConfidence;
  final String riskLevel;
  final String riskMessage;
  final String aiDiagnosis;

  AnalysisResult({
    required this.id,
    required this.timestamp,
    required this.imageBase64,
    required this.yoloDetected,
    required this.lesionType,
    required this.yoloConfidence,
    required this.mobileNetClassification,
    required this.mobileNetConfidence,
    required this.riskLevel,
    required this.riskMessage,
    required this.aiDiagnosis,
  });

  // Convert to JSON for storage
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'timestamp': timestamp.toIso8601String(),
      'imageBase64': imageBase64,
      'yoloDetected': yoloDetected,
      'lesionType': lesionType,
      'yoloConfidence': yoloConfidence,
      'mobileNetClassification': mobileNetClassification,
      'mobileNetConfidence': mobileNetConfidence,
      'riskLevel': riskLevel,
      'riskMessage': riskMessage,
      'aiDiagnosis': aiDiagnosis,
    };
  }

  // Create from JSON
  factory AnalysisResult.fromJson(Map<String, dynamic> json) {
    return AnalysisResult(
      id: json['id'] as String,
      timestamp: DateTime.parse(json['timestamp'] as String),
      imageBase64: json['imageBase64'] as String,
      yoloDetected: json['yoloDetected'] as bool,
      lesionType: json['lesionType'] as String,
      yoloConfidence: (json['yoloConfidence'] as num).toDouble(),
      mobileNetClassification: json['mobileNetClassification'] as String,
      mobileNetConfidence: (json['mobileNetConfidence'] as num).toDouble(),
      riskLevel: json['riskLevel'] as String,
      riskMessage: json['riskMessage'] as String,
      aiDiagnosis: json['aiDiagnosis'] as String,
    );
  }

  // Get image bytes from base64
  Uint8List get imageBytes => base64Decode(imageBase64);

  // Get formatted date
  String get formattedDate {
    final now = DateTime.now();
    final diff = now.difference(timestamp);
    
    if (diff.inDays == 0) {
      return 'Today, ${_formatTime(timestamp)}';
    } else if (diff.inDays == 1) {
      return 'Yesterday, ${_formatTime(timestamp)}';
    } else if (diff.inDays < 7) {
      return '${diff.inDays} days ago';
    } else {
      return '${timestamp.day}/${timestamp.month}/${timestamp.year}';
    }
  }

  String _formatTime(DateTime dt) {
    final hour = dt.hour > 12 ? dt.hour - 12 : dt.hour;
    final period = dt.hour >= 12 ? 'PM' : 'AM';
    return '$hour:${dt.minute.toString().padLeft(2, '0')} $period';
  }

  // Get risk color
  String get riskColorHex {
    switch (riskLevel) {
      case 'HIGH':
        return '#EF4444';
      case 'MODERATELY_HIGH':
        return '#F97316';
      case 'MODERATELY_LOW':
        return '#EAB308';
      case 'LOW':
      default:
        return '#22C55E';
    }
  }

  // Get readable risk level
  String get readableRiskLevel {
    switch (riskLevel) {
      case 'HIGH':
        return 'High Risk';
      case 'MODERATELY_HIGH':
        return 'Moderate-High Risk';
      case 'MODERATELY_LOW':
        return 'Moderate-Low Risk';
      case 'LOW':
      default:
        return 'No Risk';
    }
  }
}