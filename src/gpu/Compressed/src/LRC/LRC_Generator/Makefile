NVCC = nvcc
CUFLAGS = -O3
EXECUTABLE = LRC_Generator

$(EXECUTABLE): LRC_Generator.cu LRC_Generator_kernel.cu
	$(NVCC) $(CUFLAGS) LRC_Generator.cu -o $(EXECUTABLE)

clean: 
	rm -f $(EXECUTABLE)
