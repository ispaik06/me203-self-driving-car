all_imports_successful = True
print('\nLoading dependencies...\n')

# TensorFlow
try:
    import tensorflow as tf
    print(f'TensorFlow version: {tf.__version__}')
except ImportError as e:
    print("Error importing TensorFlow:", e)
    all_imports_successful = False

# OpenCV
try:
    import cv2
    print(f'OpenCV version: {cv2.__version__}')
except ImportError as e:
    print("Error importing OpenCV:", e)
    all_imports_successful = False

# Pandas
try:
    import pandas as pd
    print(f'Pandas version: {pd.__version__}')
except ImportError as e:
    print("Error importing Pandas:", e)
    all_imports_successful = False

# Numpy
try:
    import numpy as np
    print(f'Numpy version: {np.__version__}')
except ImportError as e:
    print("Error importing Numpy:", e)
    all_imports_successful = False

# Pillow
try:
    import PIL
    print(f'Pillow version: {PIL.__version__}')
except ImportError as e:
    print("Error importing Pillow:", e)
    all_imports_successful = False

# Protobuf
try:
    import google.protobuf as prtbf
    print(f'Protobuf version: {prtbf.__version__}')
except ImportError as e:
    print("Error importing Protobuf:", e)
    all_imports_successful = False

# Check if all imports are successful
if all_imports_successful:
    print("\nAll dependencies loaded!\n")
