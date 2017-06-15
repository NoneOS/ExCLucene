NVCC = nvcc
CUFLAGS = -O3
INCLUDES = -I/home/naiyong/cudpp-2.2/include
CUDPPLIB := -L/home/naiyong/cudpp-2.2/lib/ -lcudpp
EXECUTABLE = cuda_binarySearch

$(EXECUTABLE): cuda_binarySearch.cu cuda_binarySearch_kernel.cu
	$(NVCC) $(INCLUDES) $(CUDPPLIB) cuda_binarySearch.cu -o $(EXECUTABLE)

clean: 
	rm -f $(EXECUTABLE)

