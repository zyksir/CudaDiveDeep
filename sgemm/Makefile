CXX = nvcc # specify your compiler here
CUDA_PATH  = /usr/local/cuda-11.4
LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart -lcublas # specify your library linking options here
CXXFLAGS += -std=c++11 -O3 $(LDFLAGS) -I$(CUDA_PATH)/targets/x86_64-linux/include # -g -G # --ptxas-options=-v
LIBS = src/lib/*

sgemm: src/main.cu  $(LIBS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o %.cpp, $^)

clean:
	$(RM) sgemm *.prof