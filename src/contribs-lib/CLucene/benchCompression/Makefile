CXX		= /usr/share/gcc-5.4.0/bin/g++
CXXFLAGS = -std=c++11 -mcmodel=medium -march=native -mavx2 -O3 -Wno-narrowing
RM = rm
SOURCE = src
INCLUDE = include
OBJECT = object
EXECUTABLE = benchCompression
OBJS 	= $(OBJECT)/Array.o $(OBJECT)/IndexLoader.o \
		  $(OBJECT)/CodecFactory.o $(OBJECT)/LinearRegressionFactory.o \
		  $(OBJECT)/VarintGBTables.o $(OBJECT)/VarintG8IUTables.o $(OBJECT)/VarintG8CUTables.o \
		  $(OBJECT)/Simple9_Scalar.o $(OBJECT)/Simple9_SSE.o $(OBJECT)/Simple9_AVX.o \
		  $(OBJECT)/Simple16_Scalar.o $(OBJECT)/Simple16_SSE.o $(OBJECT)/Simple16_AVX.o \
		  $(OBJECT)/Unary.o

$(EXECUTABLE): $(OBJS) $(OBJECT)/benchCompression.o
	$(CXX) $(CXXFLAGS) $(OBJS) $(OBJECT)/benchCompression.o -o $(EXECUTABLE)

$(OBJECT)/benchCompression.o: benchCompressionDemo.cpp $(INCLUDE)/DeltaFactory.h $(INCLUDE)/LinearRegressionFactory.h
	$(CXX) $(CXXFLAGS) -I$(INCLUDE) -c benchCompressionDemo.cpp -o $(OBJECT)/benchCompression.o


$(OBJECT)/Array.o: $(SOURCE)/Array.cpp $(INCLUDE)/Array.h
	$(CXX) $(CXXFLAGS) -I$(INCLUDE) -c $(SOURCE)/Array.cpp -o $(OBJECT)/Array.o

$(OBJECT)/IndexLoader.o: $(SOURCE)/IndexLoader.cpp $(INCLUDE)/IndexLoader.h
	$(CXX) $(CXXFLAGS) -I$(INCLUDE) -c $(SOURCE)/IndexLoader.cpp -o $(OBJECT)/IndexLoader.o


$(OBJECT)/CodecFactory.o: $(SOURCE)/CodecFactory.cpp $(INCLUDE)/CodecFactory.h
	$(CXX) $(CXXFLAGS) -I$(INCLUDE) -c $(SOURCE)/CodecFactory.cpp -o $(OBJECT)/CodecFactory.o

$(OBJECT)/LinearRegressionFactory.o: $(SOURCE)/LinearRegressionFactory.cpp $(INCLUDE)/LinearRegressionFactory.h
	$(CXX) $(CXXFLAGS) -I$(INCLUDE) -c $(SOURCE)/LinearRegressionFactory.cpp -o $(OBJECT)/LinearRegressionFactory.o


$(OBJECT)/VarintGBTables.o: $(SOURCE)/VarintGBTables.cpp
	$(CXX) $(CXXFLAGS) -I$(INCLUDE) -c $(SOURCE)/VarintGBTables.cpp -o $(OBJECT)/VarintGBTables.o

$(OBJECT)/VarintG8IUTables.o: $(SOURCE)/VarintG8IUTables.cpp
	$(CXX) $(CXXFLAGS) -I$(INCLUDE) -c $(SOURCE)/VarintG8IUTables.cpp -o $(OBJECT)/VarintG8IUTables.o

$(OBJECT)/VarintG8CUTables.o: $(SOURCE)/VarintG8CUTables.cpp
	$(CXX) $(CXXFLAGS) -I$(INCLUDE) -c $(SOURCE)/VarintG8CUTables.cpp -o $(OBJECT)/VarintG8CUTables.o


$(OBJECT)/Simple9_Scalar.o: $(SOURCE)/Simple9_Scalar.cpp $(INCLUDE)/Simple9_Scalar.h
	$(CXX) $(CXXFLAGS) -I$(INCLUDE) -c $(SOURCE)/Simple9_Scalar.cpp -o $(OBJECT)/Simple9_Scalar.o

$(OBJECT)/Simple9_SSE.o: $(SOURCE)/Simple9_SSE.cpp $(INCLUDE)/Simple9_SSE.h
	$(CXX) $(CXXFLAGS) -I$(INCLUDE) -c $(SOURCE)/Simple9_SSE.cpp -o $(OBJECT)/Simple9_SSE.o

$(OBJECT)/Simple9_AVX.o: $(SOURCE)/Simple9_AVX.cpp $(INCLUDE)/Simple9_AVX.h
	$(CXX) $(CXXFLAGS) -I$(INCLUDE) -c $(SOURCE)/Simple9_AVX.cpp -o $(OBJECT)/Simple9_AVX.o


$(OBJECT)/Simple16_Scalar.o: $(SOURCE)/Simple16_Scalar.cpp $(INCLUDE)/Simple16_Scalar.h
	$(CXX) $(CXXFLAGS) -I$(INCLUDE) -c $(SOURCE)/Simple16_Scalar.cpp -o $(OBJECT)/Simple16_Scalar.o

$(OBJECT)/Simple16_SSE.o: $(SOURCE)/Simple16_SSE.cpp $(INCLUDE)/Simple16_SSE.h
	$(CXX) $(CXXFLAGS) -I$(INCLUDE) -c $(SOURCE)/Simple16_SSE.cpp -o $(OBJECT)/Simple16_SSE.o

$(OBJECT)/Simple16_AVX.o: $(SOURCE)/Simple16_AVX.cpp $(INCLUDE)/Simple16_AVX.h
	$(CXX) $(CXXFLAGS) -I$(INCLUDE) -c $(SOURCE)/Simple16_AVX.cpp -o $(OBJECT)/Simple16_AVX.o


$(OBJECT)/Unary.o: $(SOURCE)/Unary.cpp $(INCLUDE)/Unary.h
	$(CXX) $(CXXFLAGS) -I$(INCLUDE) -c $(SOURCE)/Unary.cpp -o $(OBJECT)/Unary.o

clean:
	$(RM) -f $(EXECUTABLE) 
	$(RM) -f $(OBJECT)/*
