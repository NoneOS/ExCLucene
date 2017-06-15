NVCC = nvcc
CUFLAGS = -O3
EXECUTABLE = SegLRC_Generator

$(EXECUTABLE): SegLRC_Generator.cu SegLRC_Generator_kernel.cu
	$(NVCC) $(CUFLAGS) SegLRC_Generator.cu -o $(EXECUTABLE)

clean: 
	rm -f $(EXECUTABLE)

