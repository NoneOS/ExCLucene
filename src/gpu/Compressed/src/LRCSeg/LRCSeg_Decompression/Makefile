NVCC = nvcc
CUFLAGS = -O3
EXECUTABLE = LRCSeg_Decompression

$(EXECUTABLE): LRCSeg_Decompression.cu LRCSeg_Decompression_kernel.cu
	$(NVCC) $(CUFLAGS) LRCSeg_Decompression.cu -o $(EXECUTABLE)

clean: 
	rm -f $(EXECUTABLE)

