BUILD_DIR = build
THIRD_PARTY_DIR = third_party

.PHONY: all
all: build
	cmake --build ${BUILD_DIR} --config Debug

.PHONY: build
build: ${THIRD_PARTY_DIR}/libtorch
	cmake -B${BUILD_DIR} -GNinja -DCMAKE_PREFIX_PATH=$(shell pwd)/${THIRD_PARTY_DIR}/libtorch .

${THIRD_PARTY_DIR}/libtorch: ${THIRD_PARTY_DIR}/libtorch-shared-with-deps-latest.zip
	cd third_party && ([ -d "libtorch" ] || unzip libtorch-shared-with-deps-latest.zip)

${THIRD_PARTY_DIR}/libtorch-shared-with-deps-latest.zip:
	mkdir -p ${THIRD_PARTY_DIR}
	wget -P third_party https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
