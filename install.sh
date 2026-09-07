cpip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch_geometric
pip install scipy tqdm IPython tensorboard matplotlib plotly pyyaml pynvml gputil trimesh glfw imgui opencv-python scikit-learn

python setup.py develop

# submodule: fairmotion (https://github.com/sunny-Codes/fairmotion)
cd src/fairmotion
python setup.py develop
