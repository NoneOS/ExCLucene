NVCC = nvcc
CUFLAGS = -O3
EXECUTABLE = ParaPFD_Decompression

$(EXECUTABLE): ParaPFD_Decompression.cu ParaPFD_Decompression_kernel.cu
	$(NVCC) $(CUFLAGS) ParaPFD_Decompression.cu -o $(EXECUTABLE)

clean: 
	rm -f $(EXECUTABLE)

