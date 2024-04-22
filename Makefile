.PHONY: all
all: build
	cmake --build build --config Debug

build: libtorch
	cmake -Bbuild -GNinja -DCMAKE_PREFIX_PATH=$(shell pwd)/libtorch .

libtorch: libtorch-shared-with-deps-latest.zip
	unzip libtorch-shared-with-deps-latest.zip

libtorch-shared-with-deps-latest.zip:
	wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
