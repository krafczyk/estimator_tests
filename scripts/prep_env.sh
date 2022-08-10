set -e

source ~/conda.sh

export wheel_arch_suffix="x86_64"
env_dir="./estimator_test_venv"

# Create base conda environment
if [ ! -e ${env_dir} ]; then
conda env create -f ./env.yaml -p ${env_dir}
fi;

# Activate
conda activate ${env_dir}

pushd /data1/matthew/Software/tf_build

# Install custom tensorflow
bash scripts/install.sh

popd

# install stuff from pip
pip install tensorboard-plugin-profile keras-tcn tensorflow-estimator==2.8.0 tensorboard==2.8
