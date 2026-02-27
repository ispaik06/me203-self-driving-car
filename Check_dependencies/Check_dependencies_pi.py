all_imports_successful = True
print("\nLoading dependencies...\n")

# Tflite-runtime
try:
    import tflite_runtime.interpreter as tflite
    print("tflite_runtime imported successfully")
except ImportError as e:
    print("Error importing tflite_runtime:", e)
    all_imports_successful = False

# OpenCV
try:
    import cv2
    print("cv2 imported successfully")
except ImportError as e:
    print("Error importing cv2:", e)
    all_imports_successful = False

# Picamera2
try:
    import picamera2
    print("picamera2 imported successfully")
except ImportError as e:
    print("Error importing picamera2:", e)
    all_imports_successful = False

# GPIOzero
try:
    import gpiozero
    print("gpiozero imported successfully")
except ImportError as e:
    print("Error importing gpiozero:", e)
    all_imports_successful = False

# Check if all imports are successful
if all_imports_successful:
    print("\nAll dependencies loaded!\n")