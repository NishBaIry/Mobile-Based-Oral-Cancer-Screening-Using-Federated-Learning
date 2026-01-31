import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'dart:typed_data';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import '../models/analysis_result.dart';
import '../services/storage_service.dart';

class AnalyzeScreen extends StatefulWidget {
  const AnalyzeScreen({super.key});

  @override
  State<AnalyzeScreen> createState() => _AnalyzeScreenState();
}

class _AnalyzeScreenState extends State<AnalyzeScreen> {
  static const platform = MethodChannel('oral_cancer_detector');

  final ImagePicker _picker = ImagePicker();
  File? _selectedImage;
  Uint8List? _annotatedImage;
  bool _isProcessing = false;
  bool _isGeminiLoading = false;
  Map<String, dynamic>? _detectionResult;
  String? _aiDiagnosis;
  
  // Model selection
  String _selectedModel = 'base';
  int _lastModelRound = 0;

  // Load Gemini API key from .env
  String get _geminiApiKey => dotenv.env['GEMINI_API_KEY'] ?? '';
  
  @override
  void initState() {
    super.initState();
    _loadModelInfo();
  }
  
  Future<void> _loadModelInfo() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _lastModelRound = prefs.getInt('last_model_round') ?? 0;
    });
  }

  Future<void> _pickFromGallery() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() {
        _selectedImage = File(image.path);
        _annotatedImage = null;
        _detectionResult = null;
        _aiDiagnosis = null;
      });
    }
  }

  Future<void> _pickFromCamera() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.camera);
    if (image != null) {
      setState(() {
        _selectedImage = File(image.path);
        _annotatedImage = null;
        _detectionResult = null;
        _aiDiagnosis = null;
      });
    }
  }

  Future<void> _processImage() async {
    if (_selectedImage == null) return;

    setState(() {
      _isProcessing = true;
      _detectionResult = null;
      _aiDiagnosis = null;
      _annotatedImage = null;
    });

    try {
      final imageBytes = await _selectedImage!.readAsBytes();

      // Step 1: Run ML detection
      final result = await platform.invokeMethod('analyzeImage', {
        'imageBytes': imageBytes,
      });

      if (result['success']) {
        final annotatedBytes = result['annotatedImage'] as Uint8List?;

        setState(() {
          _annotatedImage = annotatedBytes;
          _detectionResult = {
            'yoloDetected': result['yoloDetected'],
            'lesionType': result['lesionType'],
            'yoloConfidence': result['yoloConfidence'],
            'mobileNetClassification': result['mobileNetClassification'],
            'mobileNetConfidence': result['mobileNetConfidence'],
            'riskLevel': result['riskLevel'],
            'riskMessage': result['riskMessage'],
          };
          _isProcessing = false;
          _isGeminiLoading = true;
        });

        // Step 2: Get Gemini AI diagnosis
        final diagnosis = await _getGeminiDiagnosis(_detectionResult!);
        setState(() {
          _aiDiagnosis = diagnosis;
          _isGeminiLoading = false;
        });

        // Step 3: Save to history
        await _saveToHistory();
      } else {
        setState(() {
          _aiDiagnosis = 'Error: ${result['error']}';
          _isProcessing = false;
        });
      }
    } catch (e) {
      setState(() {
        _aiDiagnosis = 'Error during processing: $e';
        _isProcessing = false;
        _isGeminiLoading = false;
      });
    }
  }

  Future<void> _saveToHistory() async {
    if (_detectionResult == null || _annotatedImage == null) return;

    final result = AnalysisResult(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      timestamp: DateTime.now(),
      imageBase64: base64Encode(_annotatedImage!),
      yoloDetected: _detectionResult!['yoloDetected'],
      lesionType: _detectionResult!['lesionType'],
      yoloConfidence: _detectionResult!['yoloConfidence'],
      mobileNetClassification: _detectionResult!['mobileNetClassification'],
      mobileNetConfidence: _detectionResult!['mobileNetConfidence'],
      riskLevel: _detectionResult!['riskLevel'],
      riskMessage: _detectionResult!['riskMessage'],
      aiDiagnosis: _aiDiagnosis ?? '',
    );

    await StorageService.saveAnalysis(result);
  }

  Future<String> _getGeminiDiagnosis(Map<String, dynamic> detection) async {
    print('🤖 Gemini API Key loaded: ${_geminiApiKey.isNotEmpty ? "Yes (${_geminiApiKey.substring(0, 10)}...)" : "No"}');
    
    if (_geminiApiKey.isEmpty) {
      print('⚠️ No Gemini API key, using local diagnosis');
      return _generateLocalDiagnosis(detection);
    }

    try {
      final yoloDetected = detection['yoloDetected'] as bool;
      final lesionType = detection['lesionType'] as String;
      final yoloConf = detection['yoloConfidence'] as double;
      final mobileNetClass = detection['mobileNetClassification'] as String;
      final mobileNetConf = detection['mobileNetConfidence'] as double;
      final riskLevel = detection['riskLevel'] as String;
      final riskMessage = detection['riskMessage'] as String;

      final prompt = '''You are a friendly, reassuring medical AI assistant helping interpret oral health screening results.

SCREENING RESULTS:
- Lesion Detected by Scanner: ${yoloDetected ? 'Yes' : 'No'}
- Lesion Type: $lesionType
- Detection Confidence: ${(yoloConf * 100).toStringAsFixed(1)}%
- Classification: $mobileNetClass
- Classification Confidence: ${(mobileNetConf * 100).toStringAsFixed(1)}%
- System Risk Level: $riskLevel
- System Message: $riskMessage

Please provide a warm, supportive assessment in this EXACT format:

RISK ASSESSMENT: [Use the risk level: $riskLevel - convert to friendly terms like "No Risk", "Moderately Low Risk", "Moderately High Risk", or "High Risk"]

WHAT THIS MEANS:
[2-3 sentences explaining what was found in simple, non-scary terms. Be reassuring but honest.]

RECOMMENDATION:
[Clear, gentle advice on next steps. Always mention this is a screening tool, not a diagnosis.]

IMPORTANT RULES:
- Use warm, caring language
- Avoid scary medical jargon
- Don't cause unnecessary alarm
- Always recommend professional consultation for any concerning findings
- Emphasize this is just a screening tool
- COMPLETE ALL THREE SECTIONS FULLY with adequate detail
- Write 2-3 complete sentences for each section''';

      print('🌐 Calling Gemini API...');
      final response = await http
          .post(
            Uri.parse(
                'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=$_geminiApiKey'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({
              'contents': [
                {
                  'parts': [
                    {'text': prompt}
                  ]
                }
              ],
              'generationConfig': {
                'temperature': 0.7,
                'maxOutputTokens': 2048,
                'topP': 0.95,
                'topK': 40,
              },
              'safetySettings': [
                {
                  'category': 'HARM_CATEGORY_HARASSMENT',
                  'threshold': 'BLOCK_NONE'
                },
                {
                  'category': 'HARM_CATEGORY_HATE_SPEECH',
                  'threshold': 'BLOCK_NONE'
                },
                {
                  'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
                  'threshold': 'BLOCK_NONE'
                },
                {
                  'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                  'threshold': 'BLOCK_NONE'
                }
              ]
            }),
          )
          .timeout(
            const Duration(seconds: 60),
            onTimeout: () {
              print('⏱️ Gemini API timeout after 60 seconds');
              throw Exception('Request timed out');
            },
          );

      print('📡 Gemini API Response: ${response.statusCode}');

      if (response.statusCode == 200) {
        final jsonResponse = jsonDecode(response.body);
        
        // Log full response for debugging
        print('📋 Full API Response: ${response.body.substring(0, response.body.length > 500 ? 500 : response.body.length)}...');

        if (jsonResponse['candidates'] != null &&
            jsonResponse['candidates'].isNotEmpty) {
          final candidate = jsonResponse['candidates'][0];
          
          // Check finish reason
          final finishReason = candidate['finishReason'];
          print('🏁 Finish Reason: $finishReason');

          if (candidate['content'] != null &&
              candidate['content']['parts'] != null &&
              candidate['content']['parts'].isNotEmpty) {
            final text = candidate['content']['parts'][0]['text'];
            if (text != null && text.toString().isNotEmpty) {
              final responseText = text.toString();
              print('✅ Gemini API success - response length: ${responseText.length} chars');
              print('📝 Full Response:\n$responseText');
              
              // Validate all sections are present
              final hasSections = responseText.contains('RISK ASSESSMENT') &&
                                responseText.contains('WHAT THIS MEANS') &&
                                responseText.contains('RECOMMENDATION');
              if (!hasSections) {
                print('⚠️ Warning: Response incomplete - missing sections. FinishReason: $finishReason');
              }
              
              return responseText;
            }
          }

          if (candidate['finishReason'] == 'SAFETY') {
            print('⚠️ Gemini safety filter triggered');
            return _generateLocalDiagnosis(detection);
          }
        }

        print('⚠️ Gemini response structure unexpected');
        return _generateLocalDiagnosis(detection);
      } else {
        print('❌ Gemini API error: ${response.statusCode} - ${response.body}');
        return _generateLocalDiagnosis(detection);
      }
    } catch (e) {
      print('❌ Gemini API exception: $e');
      return _generateLocalDiagnosis(detection);
    }
  }

  String _generateLocalDiagnosis(Map<String, dynamic> detection) {
    final riskLevel = detection['riskLevel'] as String;
    final yoloDetected = detection['yoloDetected'] as bool;
    final lesionType = detection['lesionType'] as String;

    String riskDisplay;
    String explanation;
    String recommendation;

    switch (riskLevel) {
      case 'HIGH':
        riskDisplay = '🔴 High Risk';
        explanation =
            'Our screening has identified an area that needs professional attention. While this doesn\'t mean anything is definitely wrong, it\'s important to have it checked by a specialist.';
        recommendation =
            'We recommend scheduling an appointment with a dental professional or oral specialist soon for a proper examination.';
        break;
      case 'MODERATELY_HIGH':
        riskDisplay = '🟠 Moderately High Risk';
        explanation =
            'The screening detected an area that warrants professional review. This is a precautionary finding - many such findings turn out to be harmless.';
        recommendation =
            'Consider scheduling a checkup with your dentist to have this area examined. There\'s no immediate urgency, but professional evaluation is advised.';
        break;
      case 'MODERATELY_LOW':
        riskDisplay = '🟡 Moderately Low Risk';
        explanation =
            yoloDetected
                ? 'A $lesionType was detected and appears to be benign based on our analysis. This is generally a reassuring finding.'
                : 'The overall assessment suggests low concern, though some areas may benefit from monitoring.';
        recommendation =
            'Continue regular dental checkups. If you notice any changes or have concerns, don\'t hesitate to consult your dentist.';
        break;
      case 'LOW':
      default:
        riskDisplay = '🟢 No Risk';
        explanation =
            yoloDetected
                ? 'A $lesionType was detected but appears normal. The analysis suggests no immediate concerns.'
                : 'No specific lesions were detected. The overall assessment appears normal.';
        recommendation =
            'Maintain regular oral hygiene and routine dental visits. This screening tool is for awareness only - always consult professionals for any concerns.';
        break;
    }

    return '''RISK ASSESSMENT: $riskDisplay

WHAT THIS MEANS:
$explanation

RECOMMENDATION:
$recommendation''';
  }

  Color _getRiskColor(String? riskLevel) {
    if (riskLevel == null) return Colors.grey;
    switch (riskLevel) {
      case 'HIGH':
        return const Color(0xFFEF4444);
      case 'MODERATELY_HIGH':
        return const Color(0xFFF97316);
      case 'MODERATELY_LOW':
        return const Color(0xFFEAB308);
      case 'LOW':
      default:
        return const Color(0xFF22C55E);
    }
  }

  IconData _getRiskIcon(String? riskLevel) {
    if (riskLevel == null) return Icons.help_outline;
    switch (riskLevel) {
      case 'HIGH':
        return Icons.warning_rounded;
      case 'MODERATELY_HIGH':
        return Icons.error_outline;
      case 'MODERATELY_LOW':
        return Icons.info_outline;
      case 'LOW':
      default:
        return Icons.check_circle_outline;
    }
  }

  void _resetAnalysis() {
    setState(() {
      _selectedImage = null;
      _annotatedImage = null;
      _detectionResult = null;
      _aiDiagnosis = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8FAFC),
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios, size: 20),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text(
          'Analyze Image',
          style: TextStyle(
            fontWeight: FontWeight.w600,
            fontSize: 18,
          ),
        ),
        actions: [
          if (_detectionResult != null)
            IconButton(
              icon: const Icon(Icons.refresh),
              onPressed: _resetAnalysis,
              tooltip: 'New Analysis',
            ),
        ],
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Info Banner
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      const Color(0xFF4A90E2).withOpacity(0.1),
                      const Color(0xFF7B68EE).withOpacity(0.1),
                    ],
                  ),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                    color: const Color(0xFF4A90E2).withOpacity(0.3),
                  ),
                ),
                child: Row(
                  children: [
                    Icon(
                      Icons.lightbulb_outline,
                      color: const Color(0xFF4A90E2),
                      size: 22,
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        'Upload a clear image of the oral area for best results',
                        style: TextStyle(
                          fontSize: 13,
                          color: const Color(0xFF1E293B).withOpacity(0.8),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),
              
              // Model Selection
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                    color: const Color(0xFFE2E8F0),
                  ),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(
                          Icons.model_training,
                          color: const Color(0xFF4A90E2),
                          size: 20,
                        ),
                        const SizedBox(width: 8),
                        Text(
                          'Model Selection',
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                            color: const Color(0xFF1E293B),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12),
                      decoration: BoxDecoration(
                        color: const Color(0xFFF8FAFC),
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(
                          color: const Color(0xFFE2E8F0),
                        ),
                      ),
                      child: DropdownButtonHideUnderline(
                        child: DropdownButton<String>(
                          value: _selectedModel,
                          isExpanded: true,
                          icon: Icon(Icons.arrow_drop_down, color: const Color(0xFF64748B)),
                          style: TextStyle(
                            fontSize: 14,
                            color: const Color(0xFF1E293B),
                          ),
                          items: [
                            DropdownMenuItem(
                              value: 'base',
                              child: Text('Base Model (Original)'),
                            ),
                            if (_lastModelRound > 0)
                              DropdownMenuItem(
                                value: 'global',
                                child: Text('FL Model (Round $_lastModelRound)'),
                              ),
                          ],
                          onChanged: _isProcessing ? null : (String? newValue) {
                            if (newValue != null) {
                              setState(() {
                                _selectedModel = newValue;
                              });
                              // Optional: Show snackbar
                              ScaffoldMessenger.of(context).showSnackBar(
                                SnackBar(
                                  content: Text(
                                    newValue == 'base' 
                                        ? 'Using base model for analysis' 
                                        : 'Using FL model round $_lastModelRound',
                                  ),
                                  duration: Duration(seconds: 2),
                                  behavior: SnackBarBehavior.floating,
                                ),
                              );
                            }
                          },
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),

              // Image Selection / Display Area
              if (_annotatedImage == null && _selectedImage == null) ...[
                // Empty state - show upload options
                Container(
                  height: 260,
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: const Color(0xFFE2E8F0),
                      width: 2,
                      strokeAlign: BorderSide.strokeAlignInside,
                    ),
                  ),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: const Color(0xFFF1F5F9),
                          borderRadius: BorderRadius.circular(50),
                        ),
                        child: Icon(
                          Icons.add_photo_alternate_outlined,
                          size: 40,
                          color: Colors.grey.shade400,
                        ),
                      ),
                      const SizedBox(height: 16),
                      Text(
                        'No image selected',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w500,
                          color: Colors.grey.shade600,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Choose from gallery or take a photo',
                        style: TextStyle(
                          fontSize: 13,
                          color: Colors.grey.shade400,
                        ),
                      ),
                    ],
                  ),
                ),
              ] else ...[
                // Show selected/annotated image
                Container(
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(20),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.08),
                        blurRadius: 12,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(20),
                    child: _annotatedImage != null
                        ? Image.memory(
                            _annotatedImage!,
                            height: 260,
                            width: double.infinity,
                            fit: BoxFit.contain,
                          )
                        : Image.file(
                            _selectedImage!,
                            height: 260,
                            width: double.infinity,
                            fit: BoxFit.contain,
                          ),
                  ),
                ),
              ],
              const SizedBox(height: 16),

              // Image selection buttons
              Row(
                children: [
                  Expanded(
                    child: _ImagePickerButton(
                      icon: Icons.photo_library_outlined,
                      label: 'Gallery',
                      onTap: _isProcessing ? null : _pickFromGallery,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _ImagePickerButton(
                      icon: Icons.camera_alt_outlined,
                      label: 'Camera',
                      onTap: _isProcessing ? null : _pickFromCamera,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),

              // Analyze Button
              if (_selectedImage != null && _detectionResult == null) ...[
                ElevatedButton(
                  onPressed: _isProcessing ? null : _processImage,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF4A90E2),
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(14),
                    ),
                    elevation: 0,
                  ),
                  child: _isProcessing
                      ? Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const SizedBox(
                              width: 22,
                              height: 22,
                              child: CircularProgressIndicator(
                                color: Colors.white,
                                strokeWidth: 2.5,
                              ),
                            ),
                            const SizedBox(width: 12),
                            const Text(
                              'Analyzing...',
                              style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                            ),
                          ],
                        )
                      : Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const Icon(Icons.search, size: 22),
                            const SizedBox(width: 8),
                            const Text(
                              'Analyze Image',
                              style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                            ),
                          ],
                        ),
                ),
              ],

              // Detection Results Card
              if (_detectionResult != null) ...[
                const SizedBox(height: 8),
                _ResultCard(
                  title: 'Detection Results',
                  icon: Icons.analytics_outlined,
                  iconColor: const Color(0xFF4A90E2),
                  child: Column(
                    children: [
                      _ResultRow(
                        label: 'Lesion Detected',
                        value: _detectionResult!['yoloDetected'] ? 'Yes' : 'No',
                        valueColor: _detectionResult!['yoloDetected']
                            ? const Color(0xFFF97316)
                            : const Color(0xFF22C55E),
                      ),
                      if (_detectionResult!['yoloDetected']) ...[
                        const Divider(height: 20),
                        _ResultRow(
                          label: 'Lesion Type',
                          value: _detectionResult!['lesionType'],
                          valueColor: const Color(0xFF4A90E2),
                        ),
                        const Divider(height: 20),
                        _ResultRow(
                          label: 'Detection Confidence',
                          value: '${(_detectionResult!['yoloConfidence'] * 100).toStringAsFixed(1)}%',
                          valueColor: const Color(0xFF64748B),
                        ),
                      ],
                      const Divider(height: 20),
                      _ResultRow(
                        label: 'Classification',
                        value: _detectionResult!['mobileNetClassification'],
                        valueColor: _detectionResult!['mobileNetClassification'] == 'Malignant'
                            ? const Color(0xFFEF4444)
                            : const Color(0xFF22C55E),
                      ),
                      const Divider(height: 20),
                      _ResultRow(
                        label: 'Confidence',
                        value: '${(_detectionResult!['mobileNetConfidence'] * 100).toStringAsFixed(1)}%',
                        valueColor: const Color(0xFF64748B),
                      ),
                    ],
                  ),
                ),
              ],

              // AI Diagnosis Card
              if (_isGeminiLoading) ...[
                const SizedBox(height: 16),
                Container(
                  padding: const EdgeInsets.all(24),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(16),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.04),
                        blurRadius: 8,
                        offset: const Offset(0, 2),
                      ),
                    ],
                  ),
                  child: Column(
                    children: [
                      SizedBox(
                        width: 36,
                        height: 36,
                        child: CircularProgressIndicator(
                          strokeWidth: 3,
                          color: const Color(0xFF4A90E2),
                        ),
                      ),
                      const SizedBox(height: 14),
                      Text(
                        'Generating AI Assessment...',
                        style: TextStyle(
                          color: Colors.grey.shade500,
                          fontSize: 14,
                        ),
                      ),
                    ],
                  ),
                ),
              ] else if (_aiDiagnosis != null) ...[
                const SizedBox(height: 16),
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: _getRiskColor(_detectionResult?['riskLevel']).withOpacity(0.08),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(
                      color: _getRiskColor(_detectionResult?['riskLevel']).withOpacity(0.25),
                      width: 1.5,
                    ),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Container(
                            padding: const EdgeInsets.all(8),
                            decoration: BoxDecoration(
                              color: _getRiskColor(_detectionResult?['riskLevel']).withOpacity(0.15),
                              borderRadius: BorderRadius.circular(10),
                            ),
                            child: Icon(
                              _getRiskIcon(_detectionResult?['riskLevel']),
                              color: _getRiskColor(_detectionResult?['riskLevel']),
                              size: 20,
                            ),
                          ),
                          const SizedBox(width: 12),
                          Text(
                            'AI Health Assessment',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                              color: _getRiskColor(_detectionResult?['riskLevel']),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 14),
                      Text(
                        _aiDiagnosis!,
                        style: const TextStyle(
                          fontSize: 14,
                          height: 1.6,
                          color: Color(0xFF334155),
                        ),
                      ),
                    ],
                  ),
                ),
                
                // Save confirmation
                const SizedBox(height: 12),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                  decoration: BoxDecoration(
                    color: const Color(0xFF22C55E).withOpacity(0.1),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Row(
                    children: [
                      Icon(
                        Icons.check_circle,
                        color: const Color(0xFF22C55E),
                        size: 18,
                      ),
                      const SizedBox(width: 8),
                      Text(
                        'Result saved to history',
                        style: TextStyle(
                          color: const Color(0xFF22C55E),
                          fontSize: 13,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ],
                  ),
                ),
              ],

              const SizedBox(height: 24),

              // Bottom Disclaimer
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: const Color(0xFFF1F5F9),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(
                      Icons.medical_services_outlined,
                      color: Colors.grey.shade400,
                      size: 18,
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        'This screening tool does not replace professional medical diagnosis.',
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.grey.shade500,
                          height: 1.4,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }
}

class _ImagePickerButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback? onTap;

  const _ImagePickerButton({
    required this.icon,
    required this.label,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.white,
      borderRadius: BorderRadius.circular(12),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12),
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 14),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: const Color(0xFFE2E8F0)),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, color: const Color(0xFF4A90E2), size: 20),
              const SizedBox(width: 8),
              Text(
                label,
                style: const TextStyle(
                  color: Color(0xFF4A90E2),
                  fontWeight: FontWeight.w500,
                  fontSize: 14,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  final String title;
  final IconData icon;
  final Color iconColor;
  final Widget child;

  const _ResultCard({
    required this.title,
    required this.icon,
    required this.iconColor,
    required this.child,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: iconColor.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Icon(icon, color: iconColor, size: 20),
              ),
              const SizedBox(width: 12),
              Text(
                title,
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                  color: Color(0xFF1E293B),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          child,
        ],
      ),
    );
  }
}

class _ResultRow extends StatelessWidget {
  final String label;
  final String value;
  final Color valueColor;

  const _ResultRow({
    required this.label,
    required this.value,
    required this.valueColor,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: TextStyle(
            fontSize: 14,
            color: Colors.grey.shade600,
          ),
        ),
        Text(
          value,
          style: TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w600,
            color: valueColor,
          ),
        ),
      ],
    );
  }
}