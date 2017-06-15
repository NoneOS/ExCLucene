NVCC = nvcc
CUFLAGS = -O3
EXECUTABLE = SegLRC_Decompression

$(EXECUTABLE): SegLRC_Decompression.cu SegLRC_Decompression_kernel.cu
	$(NVCC) $(CUFLAGS) SegLRC_Decompression.cu -o $(EXECUTABLE)

clean: 
	rm -f $(EXECUTABLE)

