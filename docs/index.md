# AI on Edge Device Hardware Platforms

Example workflow for quantizing & compiling custom ML models on Edge Devices.

## Overview

EdgeAIonPlatforms demonstrates how to take trained machine-learning models and run them efficiently on resource-constrained devices such as:
HAILO, ORCA, CORAL Edge TPU and STM32 devices. 

- **Model conversion scripts** (e.g. pytorch --> ONNX, ONNX → proprietary)
- **Recommended Dockerization Strategy** so that everything is automate in a microservices format for easy scalabilty
- **Customization Instructions** to incorporate other models in the same pipeline

## Getting Started

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

#### ORCA/CORAL devices

edit the corresponding .json file

```
docker compose up -d compiler_api
```
