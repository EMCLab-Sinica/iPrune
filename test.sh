rm nvm.bin
cmake -B build -S . -D MY_DEBUG=1
make -C ./build
./build/intermittent-cnn
