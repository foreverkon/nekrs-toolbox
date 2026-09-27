PREFIX ?= $(NEKRS_HOME)
BUILD_DIR ?= build
JOBS ?= 4
CMAKE_ARGS ?=

.PHONY: all build install clean

all: build

build:
	cmake -S . -B "$(BUILD_DIR)" -DNEKRS_HOME="$(NEKRS_HOME)" \
		-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_BINDIR=bin $(CMAKE_ARGS)
	cmake --build "$(BUILD_DIR)" --parallel $(JOBS)

install: build
	test -n "$(PREFIX)"
	cmake --install "$(BUILD_DIR)" --prefix "$(abspath $(PREFIX))" --component BPTools
	cmake --install "$(BUILD_DIR)" --prefix "$(abspath $(PREFIX))" --component toolsapplications

clean:
	$(RM) -r -- "$(BUILD_DIR)"
