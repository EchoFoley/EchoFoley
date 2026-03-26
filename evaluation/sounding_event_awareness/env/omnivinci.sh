pip install -U "huggingface_hub[cli]"

hf download nvidia/omnivinci --local-dir omnivinci
cd ./omnivinci

bash ./environment_setup.sh omnivinci