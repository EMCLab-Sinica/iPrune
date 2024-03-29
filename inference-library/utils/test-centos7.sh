# dependencies
yum -y update
yum -y install gcc-c++ make cmake python3 wget
# On CentOS 7, python3 depends on older setuptools/pip - install the latest version
python3 -m pip install --user --upgrade pip setuptools
python3 -m pip install --user dataclasses numpy onnx

python3 transform.py --target msp430 --stateful --batch-size 1 mnist
mkdir build
cd build
cmake ..
make
./intermittent-cnn
