conda create -n aes python=3.10
conda activate aes
conda install -c conda-forge ffmpeg

export LD_LIBRARY_PATH="/opt/saturncloud/envs/aes/lib:${LD_LIBRARY_PATH}"

pip install audiobox_aesthetics
pip install requests
pip install torchcodec
# pip install -e .