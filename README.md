### Installation

Download the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions in the documentation. We highly recommend using a conda environment 
to simplify set up.

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# pointnet++
git clone -b drp git@github.com:Jim-Young6709/pointnet2_ops.git
pip install -e pointnet2_ops/ --no-build-isolation

# Isaac Dependencies
# first download the .tar.gz file from IsaacGym website (https://developer.nvidia.com/isaac-gym/download)
tar -xvzf IsaacGym_Preview_4_Package.tar.gz
pip install -e ./isaacgym/python/

git clone -b eef git@github.com:Jim-Young6709/IsaacGymEnvs.git
pip install -e IsaacGymEnvs/

```
