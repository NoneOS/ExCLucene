CC = g++
CCFLAGS = -std=c++0x -O3
PARAPFOR = ../../../include
OBJECT = object
EXECUTABLE = SegLRC_Compression

$(EXECUTABLE): $(OBJECT)/main.o $(OBJECT)/ParaPFor.o
	$(CC) $(CCFLAGS) $(OBJECT)/main.o $(OBJECT)/ParaPFor.o -o $(EXECUTABLE)

$(OBJECT)/main.o: main.cpp $(PARAPFOR)/ParaPFor.h
	$(CC) $(CCFLAGS) -I$(PARAPFOR) -c main.cpp -o $(OBJECT)/main.o

$(OBJECT)/ParaPFor.o: $(PARAPFOR)/ParaPFor.cpp $(PARAPFOR)/ParaPFor.h
	$(CC) $(CCFLAGS) -I$(PARAPFOR) -c $(PARAPFOR)/ParaPFor.cpp -o $(OBJECT)/ParaPFor.o


clean:
	rm -f $(EXECUTABLE) 
	rm -f $(OBJECT)/*
