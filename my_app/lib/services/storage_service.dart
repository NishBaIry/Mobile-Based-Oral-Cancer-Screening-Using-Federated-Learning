import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/analysis_result.dart';

class StorageService {
  static const String _historyKey = 'analysis_history';
  
  // Save a new analysis result
  static Future<void> saveAnalysis(AnalysisResult result) async {
    final prefs = await SharedPreferences.getInstance();
    final history = await getHistory();
    
    // Add new result at the beginning
    history.insert(0, result);
    
    // Keep only last 50 analyses to manage storage
    if (history.length > 50) {
      history.removeRange(50, history.length);
    }
    
    // Convert to JSON string list
    final jsonList = history.map((r) => jsonEncode(r.toJson())).toList();
    await prefs.setStringList(_historyKey, jsonList);
  }
  
  // Get all history
  static Future<List<AnalysisResult>> getHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final jsonList = prefs.getStringList(_historyKey) ?? [];
    
    return jsonList.map((jsonStr) {
      final json = jsonDecode(jsonStr) as Map<String, dynamic>;
      return AnalysisResult.fromJson(json);
    }).toList();
  }
  
  // Delete a specific analysis by ID
  static Future<void> deleteAnalysis(String id) async {
    final prefs = await SharedPreferences.getInstance();
    final history = await getHistory();
    
    history.removeWhere((r) => r.id == id);
    
    final jsonList = history.map((r) => jsonEncode(r.toJson())).toList();
    await prefs.setStringList(_historyKey, jsonList);
  }
  
  // Clear all history
  static Future<void> clearHistory() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_historyKey);
  }
  
  // Get history count
  static Future<int> getHistoryCount() async {
    final history = await getHistory();
    return history.length;
  }
}