NVCC = nvcc
CUFLAGS = -O3
INCLUDES = -I/home/naiyong/cudpp-2.2/include
CUDPPLIB := -L/home/naiyong/cudpp-2.2/lib/ -lcudpp
EXECUTABLE = LRCSeg_Intersection

$(EXECUTABLE): LRCSeg_Intersection.cu LRCSeg_Intersection_kernel.cu 
	$(NVCC) $(CUFLAGS) $(INCLUDES) $(CUDPPLIB) LRCSeg_Intersection.cu -o $(EXECUTABLE)

clean: 
	rm -f $(EXECUTABLE)

