NVCC = nvcc
CUFLAGS = -O3
INCLUDES = -I/home/naiyong/cudpp-2.2/include
CUDPPLIB := -L/home/naiyong/cudpp-2.2/lib/ -lcudpp
EXECUTABLE = SegLRC_Intersection

$(EXECUTABLE): SegLRC_Intersection.cu SegLRC_Intersection_kernel.cu 
	$(NVCC) $(CUFLAGS) $(INCLUDES) $(CUDPPLIB) SegLRC_Intersection.cu -o $(EXECUTABLE)

clean: 
	rm -f $(EXECUTABLE)

