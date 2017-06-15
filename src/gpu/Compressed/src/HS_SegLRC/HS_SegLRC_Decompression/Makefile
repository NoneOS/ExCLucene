NVCC = nvcc
CUFLAGS = -O3
EXECUTABLE = HS_SegLRC_Decompression

$(EXECUTABLE): HS_SegLRC_Decompression.cu HS_SegLRC_Decompression_kernel.cu
	$(NVCC) $(CUFLAGS) HS_SegLRC_Decompression.cu -o $(EXECUTABLE)

clean: 
	rm -f $(EXECUTABLE)

