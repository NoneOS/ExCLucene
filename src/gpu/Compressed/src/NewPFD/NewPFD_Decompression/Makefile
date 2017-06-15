NVCC = nvcc
CUFLAGS = -O3
EXECUTABLE = NewPFD_Decompression

$(EXECUTABLE): NewPFD_Decompression.cu NewPFD_Decompression_kernel.cu
	$(NVCC) $(CUFLAGS) NewPFD_Decompression.cu -o $(EXECUTABLE)

clean: 
	rm -f $(EXECUTABLE)

