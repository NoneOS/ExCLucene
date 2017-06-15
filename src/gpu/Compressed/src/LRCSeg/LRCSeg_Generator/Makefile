NVCC = nvcc
CUFLAGS = -O3
EXECUTABLE = LRCSeg_Generator

$(EXECUTABLE): LRCSeg_Generator.cu LRCSeg_Generator_kernel.cu
	$(NVCC) $(CUFLAGS) LRCSeg_Generator.cu -o $(EXECUTABLE)

clean: 
	rm -f $(EXECUTABLE)

