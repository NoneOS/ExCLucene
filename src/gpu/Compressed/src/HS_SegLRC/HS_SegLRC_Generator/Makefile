NVCC = nvcc
CUFLAGS = -O3
EXECUTABLE = HS_SegLRC_Generator

$(EXECUTABLE): HS_SegLRC_Generator.cu HS_SegLRC_Generator_kernel.cu
	$(NVCC) $(CUFLAGS) HS_SegLRC_Generator.cu -o $(EXECUTABLE)

clean: 
	rm -f $(EXECUTABLE)

