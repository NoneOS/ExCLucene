NVCC = nvcc
INCLUDES = -I/home/naiyong/cudpp-2.2/include
CUDPPLIB := -L/home/naiyong/cudpp-2.2/lib/ -lcudpp
CUTILLIB := -L./lib/ -lcutil_x86_64
EXECUTABLE = cuda_hashSegment 

$(EXECUTABLE): cuda_hashSegment.cu cuda_hashSegment_kernel.cu cutil_inc.h
	$(NVCC) $(INCLUDES) $(CUDPPLIB) $(CUTILLIB) cuda_hashSegment.cu -o $(EXECUTABLE)

clean: 
	rm -f $(EXECUTABLE)
	rm -f data/*

