# Jetson AGX Thor (JetPack 7, CUDA 13, SBSA). Thor runs the standard arm64 SBSA NGC containers, so
# the base is the regular NGC PyTorch image (pick a release whose CUDA matches the installed JetPack 7
# driver; check https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).
ARG BASE=nvcr.io/nvidia/pytorch:25.08-py3
FROM ${BASE}

ENV PYTHONPATH=/workspace \
    HF_HOME=/data/hf \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
        "transformers>=4.52,<5" accelerate einops timm sentencepiece \
        anthropic requests pyyaml pillow pytest \
 && (python3 -c "import cv2" 2>/dev/null || pip install --no-cache-dir opencv-python-headless) \
 && python3 -c "import torch, transformers, cv2; print('torch', torch.__version__, 'transformers', transformers.__version__)"

WORKDIR /workspace
ENTRYPOINT ["python3", "-m", "trails"]
CMD ["info"]
