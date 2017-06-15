NVCC = nvcc
CUFLAGS = -O3
INCLUDES = -I/home/naiyong/cudpp-2.2/include
CUDPPLIB := -L/home/naiyong/cudpp-2.2/lib/ -lcudpp
EXECUTABLE = LRC_Intersection

$(EXECUTABLE): LRC_Intersection.cu LRC_Intersection_kernel.cu
	$(NVCC) $(CUFLAGS) $(INCLUDES) $(CUDPPLIB) LRC_Intersection.cu -o $(EXECUTABLE)

clean: 
	rm -f $(EXECUTABLE)
