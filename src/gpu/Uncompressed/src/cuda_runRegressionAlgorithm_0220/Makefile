NVCC = nvcc
INCLUDES = -I/home/naiyong/cudpp-2.2/include
CUDPPLIB := -L/home/naiyong/cudpp-2.2/lib/ -lcudpp
CUTILLIB := -L/home/naiyong/cuda_binarySearch_rank_topk_radix_newindex_ywq/lib/ -lcutil_x86_64
EXECUTABLE = cuda_linearRegression

$(EXECUTABLE) : cuda_linearRegression.cu cuda_linearRegression_kernel.cu cutil_inc.h
	$(NVCC) $(INCLUDES) $(CUDPPLIB) $(CUTILLIB) cuda_linearRegression.cu -o $(EXECUTABLE)

clean: 
	rm -f $(EXECUTABLE)
	rm -f data/*

