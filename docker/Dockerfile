# syntax=docker/dockerfile:1
# Direct3D-S2 Docker Environment
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

LABEL name="direct3d-s2" maintainer="direct3d-s2"

# Create workspace folder and set it as working directory
RUN mkdir -p /workspace
WORKDIR /workspace

# Update package lists and install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    git-lfs \
    wget \
    vim \
    unzip \
    pkg-config \
    cmake \
    curl \
    libegl1-mesa-dev \
    libglib2.0-0 \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    mesa-utils-extra \
    libeigen3-dev \
    python3-dev \
    python3-setuptools \
    libcgal-dev \
    libsparsehash-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
ENV PYOPENGL_PLATFORM=egl

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Limit parallel compilation to avoid OOM
ENV MAX_JOBS=2

# Install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Update PATH environment variable
ENV PATH="/workspace/miniconda3/bin:${PATH}"

# Initialize conda
RUN conda init bash

# Create and activate conda environment
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n direct3d-s2 python=3.10 -y && echo "source activate direct3d-s2" > ~/.bashrc
ENV PATH /workspace/miniconda3/envs/direct3d-s2/bin:$PATH

# Set conda to always auto-approve
RUN conda config --set always_yes true

# Install conda packages
RUN conda install Ninja
RUN conda install cuda -c nvidia/label/cuda-12.1.0 -y
RUN conda install -c conda-forge libstdcxx-ng -y

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.5.1 torchvision==0.20.1 xformers --index-url https://download.pytorch.org/whl/cu121

# Clone Direct3D-S2 repository
RUN git clone https://github.com/dns64/Direct3D-S2.git
WORKDIR /workspace/Direct3D-S2

# Install Direct3D-S2 dependencies
# Install everything except voxelize and flash-attn
RUN pip install $(grep -v -e 'third_party/voxelize' -e 'flash-attn' requirements.txt | tr '\n' ' ')

# Install flash-attn and voxelize separately (both need torch visible at build time)
RUN pip install --no-build-isolation flash-attn
RUN pip install --no-build-isolation ./third_party/voxelize/

# Install Direct3D-S2 package itself
RUN pip install --no-build-isolation .

# Install additional dependencies
RUN pip install timm kornia gradio

# Try to install torchsparse using the official installer (fallback to manual compilation)
RUN git clone https://github.com/mit-han-lab/torchsparse.git /workspace/torchsparse && cd /workspace/torchsparse && FORCE_CUDA=1 pip install --no-build-isolation . && rm -rf /workspace/torchsparse

# Download model weights from HuggingFace so the image is self-contained
RUN pip install huggingface_hub && python -c "\
from huggingface_hub import hf_hub_download; \
import os; \
os.makedirs('/workspace/models/direct3d-s2-v-1-1', exist_ok=True); \
[hf_hub_download('wushuang98/Direct3D-S2', subfolder='direct3d-s2-v-1-1', filename=f, repo_type='model', local_dir='/workspace/models') \
 for f in ['config.yaml', 'model_dense.ckpt', 'model_sparse_512.ckpt', 'model_sparse_1024.ckpt', 'model_refiner.ckpt', 'model_refiner_1024.ckpt']]"

# Pre-download BiRefNet (background removal model)
RUN python -c "from transformers import AutoModelForImageSegmentation; \
AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)"

# Pre-download DINOv2 (image encoder)
RUN python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', pretrained=True)"

# Set global library paths to ensure proper linking at runtime
ENV LD_LIBRARY_PATH="/workspace/miniconda3/envs/direct3d-s2/lib:${LD_LIBRARY_PATH}"

# Activate conda environment by default
RUN echo "conda activate direct3d-s2" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Create test script to check installation
RUN cat <<'SCRIPT' > /workspace/test_installation.sh
#!/bin/bash
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate direct3d-s2
echo "Testing Direct3D-S2 installation..."
python -c "import torch; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')" || echo "PyTorch import failed"
python -c "import direct3d_s2; print('Direct3D-S2 imported successfully')" || echo "Direct3D-S2 import failed"
python -c "import torchsparse; print('TorchSparse imported successfully')" || echo "TorchSparse not available - some features may be limited"
python -c "import timm, kornia, gradio; print('Additional dependencies imported successfully')" || echo "Some dependencies missing"
echo "Installation check complete!"
SCRIPT
RUN chmod +x /workspace/test_installation.sh

# Create helper script for manual torchsparse compilation if needed
RUN cat <<'SCRIPT' > /workspace/compile_torchsparse.sh
#!/bin/bash
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate direct3d-s2
echo "Manual TorchSparse compilation..."
cd /workspace
rm -rf torchsparse
git clone https://github.com/mit-han-lab/torchsparse.git
cd torchsparse
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
python setup.py clean --all
python setup.py install
cd /workspace && rm -rf torchsparse
SCRIPT
RUN chmod +x /workspace/compile_torchsparse.sh

# Set working directory to Direct3D-S2 project
WORKDIR /workspace/Direct3D-S2

# Set default command
CMD ["/bin/bash", "-c", "echo 'Welcome to Direct3D-S2 Docker environment!' && echo 'Run /workspace/test_installation.sh to check installation status' && echo 'If TorchSparse failed, try /workspace/compile_torchsparse.sh' && /bin/bash"]
