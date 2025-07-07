# EdgeAIonPlatforms

A collection of tools, examples, and best practices for deploying AI/ML models on edge devices and IoT platforms.

## Overview

EdgeAIonPlatforms demonstrates how to take trained machine-learning models and run them efficiently on resource-constrained devices such as Raspberry Pi, NVIDIA Jetson, Coral USB TPU, and Intel Neural Compute Stick. Each platform folder contains:

- **Model conversion scripts** (e.g. TensorFlow → TFLite, ONNX → OpenVINO)
- **Runtime examples** in Python or C++
- **Performance benchmarks** and optimization tips

## Getting Started

- A compatible edge device (see **Supported Platforms** below)

### Installation

1. **Clone the repo**  
   ```
   git clone https://github.com/alema416/EdgeAIonPlatforms.git
   cd EdgeAIonPlatforms
   ```

2. **Build/Pull the Docker Images**
   ```
   docker compose build stm32ai
   docker compose pull degirum_api hailo
   ```

### Usage

#### HAILO

edit configs/optimizer.yaml

```
docker compose run --rm --entrypoint /usr/bin/bash hailo
```

#### ST Devices

edit configs/quantization_config.yaml
```
docker compose run --rm stm32ai
```

edit the corresponding .json file
#### ORCA/CORAL devices

```
docker compose up -d compiler_api
```
