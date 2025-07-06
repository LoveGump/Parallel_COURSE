CXX = g++
CXXFLAGS = -O3 -march=native -fopenmp -std=c++17 -Wall -Wextra
INCLUDES = -I.
SOURCES = main.cc simd_openmp_ntt.cpp
TARGET = ntt_simd_openmp

# 创建files目录（如果不存在）
DIRS = files

$(TARGET): $(DIRS) $(SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET)

$(DIRS):
	mkdir -p $(DIRS)

clean:
	rm -f $(TARGET)
	rm -rf $(DIRS)

run: $(TARGET)
	./$(TARGET)

.PHONY: clean run $(DIRS)
