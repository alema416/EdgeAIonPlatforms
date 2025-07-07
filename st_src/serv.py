#!/usr/bin/env python3

# first run: sudo ip route add 192.168.1.11/32 via 192.168.1.1 dev wlp2s0
import os
import sys
import json
from flask import Flask, request, jsonify
from datetime import datetime
import subprocess
import base64
import os
from hydra import initialize, compose
import base64
import requests
with initialize(config_path="../configs/"):
    cfg = compose(config_name="base")  # exp1.yaml with defaults key
#!/usr/bin/env python3
import os
import argparse
from collections import OrderedDict

# 1) Import your modified ResNet18 (with torch.flatten)


import subprocess
import re

def get_quantized_models_dir(run_id: int,
    stm32ai_script: str,
    config_path: str,
    config_name: str,
    model_path: str
) -> str:
    """
    Runs the stm32ai_main.py quantization command and returns
    the path to the generated 'quantized_models' directory.
    
    :param stm32ai_script: full path to stm32ai_main.py
    :param config_path:     --config-path argument
    :param config_name:     --config-name argument
    :param model_path:      general.model_path=... argument
    :return: the directory path where quantized models were saved
    :raises RuntimeError: if the directory path cannot be found in output
    """
    cmd = [
        "python3",
        stm32ai_script,
        "--config-path", config_path,
        "--config-name", config_name,
        f"general.model_path={model_path}"
    ]
    print(cmd)
    # run and capture both stdout and stderr as text
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False
    )
    
    output = proc.stdout
    print(output)
    # look for the line containing "Model available at : <path>"
    m = re.search(r"Model available at\s*:\s*(\S+)", output)
    dirs = re.findall(r"(/[^\s]+/quantized_models)", output)

    if not dirs:
        raise RuntimeError(
            "Could not find 'quantized_models' directory in stm32ai_main output"
        )
    with open("output.txt", "a") as f:
        # convert num to str, join with your text, and end with a newline
        f.write(str(run_id) + " " + dirs[-1] + "\n")
    return dirs[-1] #m.group(1)


if __name__ == '__main__':
    script = "/data/pgdata/stm32ai-modelzoo-services/image_classification/stm32ai_main.py"
    cfg_path = "/data/configs/"
    cfg_name = "quantization_config"
    mdl_path = f"/data/models/test/0/model_32_opset17_quant_qdq_pc.onnx"
    run_id = 0
    print(get_quantized_models_dir(run_id, script, cfg_path, cfg_name, mdl_path))
