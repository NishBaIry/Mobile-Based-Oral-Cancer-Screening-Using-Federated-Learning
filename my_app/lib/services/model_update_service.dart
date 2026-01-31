import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

class ModelUpdateService {
  static const platform = MethodChannel('com.example.my_app/model');
  
  static String? _serverHost;
  static String? _serverPort;
  static bool _autoUpdate = true;
  
  // Initialize service
  static Future<void> initialize() async {
    _serverHost = dotenv.env['FL_SERVER_HOST'] ?? '192.168.1.100';
    _serverPort = dotenv.env['FL_SERVER_PORT'] ?? '5000';
    _autoUpdate = dotenv.env['AUTO_UPDATE_MODEL']?.toLowerCase() == 'true';
    
    print('🔄 Model Update Service initialized');
    print('   Server: http://$_serverHost:$_serverPort');
    print('   Auto-update: $_autoUpdate');
  }
  
  // Check for model updates
  static Future<bool> checkForUpdates() async {
    try {
      final url = Uri.parse('http://$_serverHost:$_serverPort/api/status');
      
      print('🔍 Checking for model updates...');
      
      final response = await http.get(url).timeout(
        const Duration(seconds: 5),
        onTimeout: () {
          print('⚠️ Connection timeout - server may be offline');
          return http.Response('Timeout', 408);
        },
      );
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        final currentRound = data['current_round'] ?? 0;
        
        // Get last known round
        final prefs = await SharedPreferences.getInstance();
        final lastRound = prefs.getInt('last_model_round') ?? 0;
        
        print('   Server round: $currentRound');
        print('   Local round: $lastRound');
        
        if (currentRound > lastRound) {
          print('✅ New model available!');
          return true;
        } else {
          print('✅ Model is up to date');
          return false;
        }
      } else {
        print('⚠️ Server returned status: ${response.statusCode}');
        return false;
      }
    } catch (e) {
      print('❌ Error checking for updates: $e');
      return false;
    }
  }
  
  // Download and update model
  static Future<bool> downloadAndUpdateModel() async {
    try {
      // Request TFLite format directly from server
      final url = Uri.parse('http://$_serverHost:$_serverPort/api/download_global_model?format=tflite');
      
      print('📥 Downloading global model (TFLite format)...');
      
      final response = await http.get(url).timeout(
        const Duration(seconds: 30),
      );
      
      if (response.statusCode == 200) {
        final modelRound = response.headers['x-model-round'] ?? 'unknown';
        final modelFormat = response.headers['x-model-format'] ?? 'unknown';
        print('   Downloaded ${response.bodyBytes.length} bytes (Round $modelRound, Format: $modelFormat)');
        
        // Send TFLite model bytes directly to native code
        final result = await platform.invokeMethod('updateModel', {
          'modelBytes': response.bodyBytes,
          'modelRound': modelRound,
        });
        
        if (result == true) {
          // Save the new round number
          final statusUrl = Uri.parse('http://$_serverHost:$_serverPort/api/status');
          final statusResponse = await http.get(statusUrl);
          
          if (statusResponse.statusCode == 200) {
            final data = json.decode(statusResponse.body);
            final currentRound = data['current_round'] ?? 0;
            
            final prefs = await SharedPreferences.getInstance();
            await prefs.setInt('last_model_round', currentRound);
            
            print('✅ Model updated to round $currentRound');
            return true;
          }
        }
        
        return false;
      } else {
        print('❌ Download failed: ${response.statusCode}');
        return false;
      }
    } catch (e) {
      print('❌ Error downloading model: $e');
      return false;
    }
  }
  
  // Check and auto-update if needed
  static Future<bool> checkAndUpdate() async {
    if (!_autoUpdate) {
      print('⚠️ Auto-update disabled');
      return false;
    }
    
    try {
      // Check if new model is available
      final hasUpdate = await checkForUpdates();
      
      if (hasUpdate) {
        print('🔄 Auto-updating model...');
        return await downloadAndUpdateModel();
      }
      
      return false;
    } catch (e) {
      print('❌ Auto-update failed: $e');
      return false;
    }
  }
  
  // Get server status
  static Future<Map<String, dynamic>?> getServerStatus() async {
    try {
      final url = Uri.parse('http://$_serverHost:$_serverPort/api/status');
      final response = await http.get(url).timeout(const Duration(seconds: 5));
      
      if (response.statusCode == 200) {
        return json.decode(response.body);
      }
      return null;
    } catch (e) {
      return null;
    }
  }
  
  // Get list of available model versions
  static Future<List<Map<String, dynamic>>> getAvailableModels() async {
    final prefs = await SharedPreferences.getInstance();
    final List<String> modelVersionsJson = prefs.getStringList('model_versions') ?? [];
    
    return modelVersionsJson.map((json) => jsonDecode(json) as Map<String, dynamic>).toList();
  }
  
  // Save model version info
  static Future<void> saveModelVersion(int round, int timestamp) async {
    final prefs = await SharedPreferences.getInstance();
    final List<String> modelVersionsJson = prefs.getStringList('model_versions') ?? [];
    
    final modelInfo = {
      'round': round,
      'timestamp': timestamp,
      'downloaded_at': DateTime.now().millisecondsSinceEpoch,
    };
    
    modelVersionsJson.add(jsonEncode(modelInfo));
    await prefs.setStringList('model_versions', modelVersionsJson);
  }
  
  // Get current active model round
  static Future<int> getCurrentModelRound() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getInt('last_model_round') ?? 0;
  }
}
