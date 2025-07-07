# Pipeline Customization for STM32 Devices

Sample config:

```
  "model_name": "model_11",
  "model_version": 1,
  "image_width": 224,
  "image_height": 224,
  "OutputNMSThreshold": "0.6",
  "calibration_images": "",
  "MaxDetectionsPerClass": "100",
  "OutputConfThreshold": "0.3",
  "MaxDetections": "100",
  "InputQuantEn": true,
  "MaxClassesPerDetection": "2",
  "input_pad_method": "stretch",
  "OutputSoftmaxEn": true,
  "input_resize_method": "bilinear",
  "input_img_norm_enabled": true,
  "input_norm_mean": "[0.50463295, 0.46120012, 0.4291694 ]",
  "input_norm_std": "[0.18364702, 0.1885083,  0.19882548]",
  "model_format": "onnx-fp32",
  "device_type": [
    "tflite-edgetpu-quant"
  ],
  "upload_zip_cloud": false,
  "output_postprocess_type": "Classification",
  "calib_img_path": "../../output_files/data/processed/IDID_broken_224/val/",
  "num_calib_imgs": 256,
  "separate_outputs": true,
  "UseRegularNMS": true
```

And the process can be customized by trial & error (no documentation available as per July 2025).
