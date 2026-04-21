@echo off
echo Setting up local environment for VGGT Pipeline...

echo 1. Cloning VGGT repository...
if not exist vggt (
    git clone https://github.com/facebookresearch/vggt.git
) else (
    echo VGGT directory already exists, skipping clone.
)

cd vggt

echo 2. Installing requirements...
pip uninstall -y pycolmap pyceres numpy
pip install numpy==1.26.4
pip install -r requirements.txt
pip install pycolmap==3.10.0 pyceres==2.3
pip install hydra-core trimesh open3d huggingface_hub plotly

echo 3. Installing PyTorch with CUDA 11.8 (Adjust if your system needs a different CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo 4. Installing specific LightGlue fork...
if not exist LightGlue (
    git clone https://github.com/cvg/LightGlue.git
)
cd LightGlue
python -m pip install -e .
cd ..

echo Setup complete.
cd ..
pause
