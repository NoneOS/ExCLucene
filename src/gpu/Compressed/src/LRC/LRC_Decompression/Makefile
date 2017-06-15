NVCC = nvcc
CUFLAGS = -O3
EXECUTABLE = LRC_Decompression

$(EXECUTABLE): LRC_Decompression.cu LRC_Decompression_kernel.cu
	$(NVCC) $(CUFLAGS) LRC_Decompression.cu -o $(EXECUTABLE)

clean: 
	rm -f $(EXECUTABLE)

