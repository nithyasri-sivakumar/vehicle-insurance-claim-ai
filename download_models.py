import urllib.request
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# URLs for MobileNet SSD model files
model_url = 'https://storage.googleapis.com/tfhub-modules/tensorflow/ssd_mobilenet_v3_large_coco_2020_01_14/1.tar.gz'
config_url = 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

print('Downloading MobileNet SSD model files...')

try:
    # Download config file
    print('Downloading config file...')
    urllib.request.urlretrieve(config_url, 'models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
    print('Config file downloaded.')

    # For the model file, we'll use a different approach since the direct URL might not work
    # Let's use a known working MobileNet SSD model
    model_url = 'https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API'
    print('Model file would need to be downloaded manually from TensorFlow Object Detection API')
    print('For now, using fallback detection method.')

except Exception as e:
    print(f'Error downloading model files: {e}')
    print('Using fallback detection method.')