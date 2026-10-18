CXX       := g++
CXXFLAGS  := -std=c++17 -O3 -march=native -DNDEBUG -Wall -Wextra -Icpp_core/include
BUILD_DIR := build
BIN       := $(BUILD_DIR)/main_benchmark

.PHONY: all build test plots reproduce clean

all: reproduce

build:
	@mkdir -p $(BUILD_DIR)
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) -j

test: build
	MPLCONFIGDIR=/tmp python3 -m pytest tests/ -v

plots: build
	MPLCONFIGDIR=/tmp python3 scripts/reproduce_all_figures.py

reproduce:
	bash scripts/run_full_suite.sh

clean:
	rm -rf $(BUILD_DIR) .pytest_cache __pycache__ python_stochastics/__pycache__ tests/__pycache__
