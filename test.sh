rm nvm.bin
cmake -B build -S . -D MY_DEBUG=3 -D USE_PROTOBUF=OFF
make -C ./build
./build/intermittent-cnn -r 1
