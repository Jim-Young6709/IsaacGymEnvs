### Installation

Download the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions in the documentation. We highly recommend using a conda environment 
to simplify set up.

```bash
# install torch
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# install IsaacGym
tar -xzf IsaacGym_Preview_4_Package.tar.gz
pip install -e ./isaacgym/python/

# curobo for IK
git clone https://github.com/NVlabs/curobo.git
pip install -e curobo/ --no-build-isolation

# pointnet++
git clone -b dex git@github.com:Jim-Young6709/pointnet2_ops.git
pip install -e pointnet2_ops/ --no-build-isolation

# Isaac Dependencies
# first download the .tar.gz file from IsaacGym website (https://developer.nvidia.com/isaac-gym/download)
tar -xvzf IsaacGym_Preview_4_Package.tar.gz
pip install -e ./isaacgym/python/

git clone -b eef git@github.com:Jim-Young6709/IsaacGymEnvs.git
pip install -e IsaacGymEnvs/

pip install tqdm ipdb geometrout==0.0.3.4 numpy==1.23.0 open3d urchin h5py

```

Note that if you encounter the following errors when running Isaac Gym:
```bash
in import_module return _bootstrap._gcd_import(name[level:], package, level)
  ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
```
you should do the following:
```bash
# inside (dex_drp)
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

# add on activate
cat > "$CONDA_PREFIX/etc/conda/activate.d/zz_libpath.sh" <<'EOF'
export _OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EOF

# restore on deactivate
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/zz_libpath.sh" <<'EOF'
export LD_LIBRARY_PATH="$_OLD_LD_LIBRARY_PATH"
unset _OLD_LD_LIBRARY_PATH
EOF
```
