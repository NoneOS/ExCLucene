NVCC = nvcc
CUFLAGS = -O3
INCLUDES = -I/home/naiyong/cudpp-2.2/include
CUDPPLIB := -L/home/naiyong/cudpp-2.2/lib/ -lcudpp
EXECUTABLE = HS_SegLRC_Intersection

$(EXECUTABLE): HS_SegLRC_Intersection.cu HS_SegLRC_Intersection_kernel.cu 
	$(NVCC) $(CUFLAGS) $(INCLUDES) $(CUDPPLIB) HS_SegLRC_Intersection.cu -o $(EXECUTABLE)

clean: 
	rm -f $(EXECUTABLE)

