
rm nvm.bin
cmake -B build -S .
make -C ./build
./build/intermittent-cnn > /tmp/log.txt
vim /tmp/log.txt
