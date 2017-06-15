NVCC = nvcc
CUFLAGS = -O3
INCLUDES = -I/home/naiyong/cudpp-2.2/include
CUDPPLIB := -L/home/naiyong/cudpp-2.2/lib/ -lcudpp
EXECUTABLE = ParaPFD_Intersection

$(EXECUTABLE): ParaPFD_Intersection.cu ParaPFD_Intersection_kernel.cu
	$(NVCC) $(CUFLAGS) $(INCLUDES) $(CUDPPLIB) ParaPFD_Intersection.cu -o $(EXECUTABLE)

clean: 
	rm -f $(EXECUTABLE)

