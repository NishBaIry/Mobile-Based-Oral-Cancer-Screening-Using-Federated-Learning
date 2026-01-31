package com.example.my_app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min
import android.os.Handler
import android.os.Looper

class MainActivity: FlutterActivity() {
    private val CHANNEL = "oral_cancer_detector"
    private val MODEL_CHANNEL = "com.example.my_app/model"
    private var yoloInterpreter: Interpreter? = null
    private var mobileNetInterpreter: Interpreter? = null
    private val updateCheckHandler = Handler(Looper.getMainLooper())
    private val UPDATE_CHECK_INTERVAL = 5 * 60 * 1000L // 5 minutes
    private var flutterMethodChannel: MethodChannel? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        
        try {
            yoloInterpreter = Interpreter(loadModelFile("yolo_oral_cancer.tflite"))
            mobileNetInterpreter = Interpreter(loadModelFile("mobilenetv2_oral_cancer.tflite"))
            android.util.Log.d("OralCancer", "✅ Models loaded successfully")
        } catch (e: Exception) {
            android.util.Log.e("OralCancer", "❌ Failed to load models: ${e.message}")
            e.printStackTrace()
        }

        // Main analysis channel
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            if (call.method == "analyzeImage") {
                val imageBytes = call.argument<ByteArray>("imageBytes")
                if (imageBytes != null) {
                    try {
                        val analysisResult = analyzeImage(imageBytes)
                        result.success(analysisResult)
                    } catch (e: Exception) {
                        android.util.Log.e("OralCancer", "Error during analysis: ${e.message}")
                        result.success(mapOf(
                            "success" to false,
                            "error" to e.message
                        ))
                    }
                } else {
                    result.success(mapOf(
                        "success" to false,
                        "error" to "No image data"
                    ))
                }
            } else {
                result.notImplemented()
            }
        }
        
        // Model update channel
        flutterMethodChannel = MethodChannel(flutterEngine.dartExecutor.binaryMessenger, MODEL_CHANNEL)
        flutterMethodChannel!!.setMethodCallHandler { call, result ->
            if (call.method == "updateModel") {
                val modelBytes = call.argument<ByteArray>("modelBytes")
                if (modelBytes != null) {
                    try {
                        val success = updateMobileNetModel(modelBytes)
                        result.success(success)
                    } catch (e: Exception) {
                        android.util.Log.e("OralCancer", "Error updating model: ${e.message}")
                        result.success(false)
                    }
                } else {
                    result.success(false)
                }
            } else {
                result.notImplemented()
            }
        }
        
        // Start periodic model update checking
        startPeriodicUpdateCheck()
    }
    
    private fun startPeriodicUpdateCheck() {
        updateCheckHandler.postDelayed(object : Runnable {
            override fun run() {
                android.util.Log.d("OralCancer", "🔄 Periodic check for model updates...")
                // Notify Flutter to check for updates
                flutterMethodChannel?.invokeMethod("checkForUpdates", null)
                updateCheckHandler.postDelayed(this, UPDATE_CHECK_INTERVAL)
            }
        }, UPDATE_CHECK_INTERVAL)
        android.util.Log.d("OralCancer", "✅ Periodic update check started (every ${UPDATE_CHECK_INTERVAL/60000} minutes)")
    }
    
    private fun updateMobileNetModel(modelBytes: ByteArray): Boolean {
        return try {
            android.util.Log.d("OralCancer", "📥 Updating MobileNetV2 model (${modelBytes.size} bytes)...")
            
            // Save the new model to internal storage
            val modelFile = java.io.File(applicationContext.filesDir, "mobilenetv2_oral_cancer.tflite")
            modelFile.writeBytes(modelBytes)
            
            android.util.Log.d("OralCancer", "💾 Model saved to: ${modelFile.absolutePath}")
            
            // Close old interpreter
            mobileNetInterpreter?.close()
            
            // Load new model from internal storage
            val fileInputStream = java.io.FileInputStream(modelFile)
            val fileChannel = fileInputStream.channel
            val buffer = fileChannel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, 0, fileChannel.size())
            fileChannel.close()
            fileInputStream.close()
            
            mobileNetInterpreter = Interpreter(buffer)
            
            android.util.Log.d("OralCancer", "✅ MobileNetV2 model updated successfully!")
            true
        } catch (e: Exception) {
            android.util.Log.e("OralCancer", "❌ Failed to update model: ${e.message}")
            e.printStackTrace()
            false
        }
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun analyzeImage(imageBytes: ByteArray): Map<String, Any> {
        val originalBitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            ?: return mapOf("success" to false, "error" to "Failed to decode image")

        val bitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
        originalBitmap.recycle()

        android.util.Log.d("OralCancer", "Image decoded: ${bitmap.width}x${bitmap.height}")

        // Step 1: Run YOLO detection - get BEST detection only
        val bestDetection = runYOLODetection(bitmap)
        
        // Create annotated image
        val annotatedBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(annotatedBitmap)
        
        // Paint for bounding box
        val boxPaint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = 6f
            color = Color.parseColor("#FF6B6B") // Coral red color
        }
        
        val cornerPaint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = 8f
            color = Color.parseColor("#FF6B6B")
        }

        if (bestDetection == null) {
            // No YOLO detection - run MobileNet on full image as fallback
            android.util.Log.d("OralCancer", "No YOLO detection, running MobileNet on full image")
            
            val classification = classifyWithMobileNet(bitmap)
            val mobileNetConf = classification["confidence"] as Double
            val mobileNetClass = classification["className"] as String
            
            // Determine risk level for no-detection case
            val riskLevel: String
            val riskMessage: String
            
            if (mobileNetClass == "Malignant" && mobileNetConf > 0.80) {
                riskLevel = "MODERATELY_HIGH"
                riskMessage = "Recommend checkup"
            } else {
                riskLevel = "LOW"
                riskMessage = "Monitor regularly"
            }
            
            // Convert bitmap to bytes
            val outputStream = ByteArrayOutputStream()
            annotatedBitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
            val annotatedBytes = outputStream.toByteArray()
            
            return mapOf(
                "success" to true,
                "yoloDetected" to false,
                "annotatedImage" to annotatedBytes,
                "lesionType" to "No specific lesion detected",
                "yoloConfidence" to 0.0,
                "mobileNetClassification" to mobileNetClass,
                "mobileNetConfidence" to mobileNetConf,
                "riskLevel" to riskLevel,
                "riskMessage" to riskMessage
            )
        }

        // YOLO detected a lesion - process it
        val bbox = bestDetection["bbox"] as List<Int>
        val x1 = bbox[0]
        val y1 = bbox[1]
        val x2 = bbox[2]
        val y2 = bbox[3]
        val yoloConf = bestDetection["confidence"] as Float
        val lesionType = bestDetection["className"] as String

        android.util.Log.d("OralCancer", "Best YOLO detection: $lesionType at [$x1,$y1,$x2,$y2] conf=$yoloConf")

        // Draw bounding box on image
        canvas.drawRect(x1.toFloat(), y1.toFloat(), x2.toFloat(), y2.toFloat(), boxPaint)
        
        // Draw corner accents for better visibility
        val cornerLength = 30f
        // Top-left
        canvas.drawLine(x1.toFloat(), y1.toFloat(), x1.toFloat() + cornerLength, y1.toFloat(), cornerPaint)
        canvas.drawLine(x1.toFloat(), y1.toFloat(), x1.toFloat(), y1.toFloat() + cornerLength, cornerPaint)
        // Top-right
        canvas.drawLine(x2.toFloat(), y1.toFloat(), x2.toFloat() - cornerLength, y1.toFloat(), cornerPaint)
        canvas.drawLine(x2.toFloat(), y1.toFloat(), x2.toFloat(), y1.toFloat() + cornerLength, cornerPaint)
        // Bottom-left
        canvas.drawLine(x1.toFloat(), y2.toFloat(), x1.toFloat() + cornerLength, y2.toFloat(), cornerPaint)
        canvas.drawLine(x1.toFloat(), y2.toFloat(), x1.toFloat(), y2.toFloat() - cornerLength, cornerPaint)
        // Bottom-right
        canvas.drawLine(x2.toFloat(), y2.toFloat(), x2.toFloat() - cornerLength, y2.toFloat(), cornerPaint)
        canvas.drawLine(x2.toFloat(), y2.toFloat(), x2.toFloat(), y2.toFloat() - cornerLength, cornerPaint)

        // Crop and classify with MobileNet
        val cropX = max(0, min(x1, bitmap.width - 1))
        val cropY = max(0, min(y1, bitmap.height - 1))
        val cropW = max(1, min(x2 - x1, bitmap.width - cropX))
        val cropH = max(1, min(y2 - y1, bitmap.height - cropY))

        val croppedBitmap = Bitmap.createBitmap(bitmap, cropX, cropY, cropW, cropH)
        val classification = classifyWithMobileNet(croppedBitmap)
        
        val mobileNetConf = classification["confidence"] as Double
        val mobileNetClass = classification["className"] as String

        android.util.Log.d("OralCancer", "MobileNet result: $mobileNetClass ($mobileNetConf)")

        // Determine risk level based on the logic
        val riskLevel: String
        val riskMessage: String

        if (mobileNetClass == "Malignant") {
            when {
                mobileNetConf > 0.80 -> {
                    riskLevel = "HIGH"
                    riskMessage = "Please see a doctor soon"
                }
                mobileNetConf >= 0.50 -> {
                    riskLevel = "MODERATELY_HIGH"
                    riskMessage = "Recommend checkup"
                }
                else -> {
                    // <50% malignant means leaning towards benign
                    riskLevel = "LOW"
                    riskMessage = "Lesion detected but appears benign - monitor and consult if concerned"
                }
            }
        } else {
            // Benign
            when {
                mobileNetConf > 0.80 -> {
                    riskLevel = "LOW"
                    riskMessage = "Appears normal, monitor regularly"
                }
                else -> {
                    riskLevel = "MODERATELY_LOW"
                    riskMessage = "Likely benign, consider checkup if concerned"
                }
            }
        }

        // Convert annotated bitmap to bytes
        val outputStream = ByteArrayOutputStream()
        annotatedBitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
        val annotatedBytes = outputStream.toByteArray()

        return mapOf(
            "success" to true,
            "yoloDetected" to true,
            "annotatedImage" to annotatedBytes,
            "lesionType" to lesionType,
            "yoloConfidence" to yoloConf.toDouble(),
            "mobileNetClassification" to mobileNetClass,
            "mobileNetConfidence" to mobileNetConf,
            "riskLevel" to riskLevel,
            "riskMessage" to riskMessage
        )
    }

    private fun runYOLODetection(bitmap: Bitmap): Map<String, Any>? {
        val resized = Bitmap.createScaledBitmap(bitmap, 640, 640, true)

        val inputBuffer = ByteBuffer.allocateDirect(1 * 640 * 640 * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(640 * 640)
        resized.getPixels(pixels, 0, 640, 0, 0, 640, 640)

        for (pixel in pixels) {
            val r = (pixel shr 16 and 0xFF) / 255.0f
            val g = (pixel shr 8 and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            
            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }

        // Output shape: [1, 7, 8400]
        val outputBuffer = Array(1) { Array(7) { FloatArray(8400) } }

        try {
            yoloInterpreter?.run(inputBuffer, outputBuffer)
        } catch (e: Exception) {
            android.util.Log.e("OralCancer", "YOLO inference error: ${e.message}")
            resized.recycle()
            return null
        }

        // Find the BEST detection (highest confidence) above threshold
        val confThreshold = 0.40f  // 40% threshold as specified
        var bestConf = 0f
        var bestIdx = -1
        var bestClassId = 0

        for (i in 0 until 8400) {
            val class1Conf = outputBuffer[0][5][i]  // Erythroplakia
            val class2Conf = outputBuffer[0][6][i]  // Leukoplakia
            
            val maxConf = maxOf(class1Conf, class2Conf)
            
            if (maxConf > confThreshold && maxConf > bestConf) {
                bestConf = maxConf
                bestIdx = i
                bestClassId = if (class1Conf > class2Conf) 1 else 2
            }
        }

        resized.recycle()

        if (bestIdx == -1) {
            android.util.Log.d("OralCancer", "No detection above threshold $confThreshold")
            return null
        }

        // Get bbox for best detection
        val rawCx = outputBuffer[0][0][bestIdx]
        val rawCy = outputBuffer[0][1][bestIdx]
        val rawW = outputBuffer[0][2][bestIdx]
        val rawH = outputBuffer[0][3][bestIdx]

        // Log raw values for debugging
        android.util.Log.d("OralCancer", "Raw YOLO output: cx=$rawCx, cy=$rawCy, w=$rawW, h=$rawH")

        val scaleX = bitmap.width.toFloat()
        val scaleY = bitmap.height.toFloat()

        // Check if values are normalized (0-1) or in pixels (0-640)
        val isNormalized = rawCx <= 1.0f && rawCy <= 1.0f && rawW <= 1.0f && rawH <= 1.0f

        val cx: Float
        val cy: Float
        val w: Float
        val h: Float

        if (isNormalized) {
            cx = rawCx * scaleX
            cy = rawCy * scaleY
            w = rawW * scaleX
            h = rawH * scaleY
        } else {
            cx = rawCx * (scaleX / 640f)
            cy = rawCy * (scaleY / 640f)
            w = rawW * (scaleX / 640f)
            h = rawH * (scaleY / 640f)
        }

        val x1 = max(0, (cx - w / 2).toInt())
        val y1 = max(0, (cy - h / 2).toInt())
        val x2 = min(bitmap.width, (cx + w / 2).toInt())
        val y2 = min(bitmap.height, (cy + h / 2).toInt())

        val className = if (bestClassId == 1) "Erythroplakia" else "Leukoplakia"

        android.util.Log.d("OralCancer", "Best detection: $className at [$x1,$y1,$x2,$y2] conf=$bestConf")

        return mapOf(
            "bbox" to listOf(x1, y1, x2, y2),
            "confidence" to bestConf,
            "classId" to bestClassId,
            "className" to className
        )
    }

    private fun classifyWithMobileNet(bitmap: Bitmap): Map<String, Any> {
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

        val inputBuffer = ByteBuffer.allocateDirect(1 * 224 * 224 * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(224 * 224)
        resized.getPixels(pixels, 0, 224, 0, 0, 224, 224)

        for (pixel in pixels) {
            val r = (pixel shr 16 and 0xFF) / 255.0f
            val g = (pixel shr 8 and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            
            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }

        val outputBuffer = Array(1) { FloatArray(1) }
        mobileNetInterpreter?.run(inputBuffer, outputBuffer)

        val rawOutput = outputBuffer[0][0]
        
        android.util.Log.d("OralCancer", "MobileNet raw output: $rawOutput")

        val isMalignant = rawOutput > 0.5f
        val confidence = if (isMalignant) rawOutput else (1.0f - rawOutput)
        val className = if (isMalignant) "Malignant" else "Benign"

        return mapOf(
            "className" to className,
            "confidence" to confidence.toDouble()
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        updateCheckHandler.removeCallbacksAndMessages(null)
        yoloInterpreter?.close()
        mobileNetInterpreter?.close()
    }
}