CC = g++
CCFLAGS = -std=c++0x -O3
PARAPFOR = ../../../include/
OBJECT = object
EXECUTABLE = ParaPFD_Compression

$(EXECUTABLE): $(OBJECT)/main.o $(OBJECT)/ParaPFor.o
	$(CC) $(CCFLAGS) -I$(PARAPFOR) $(OBJECT)/main.o $(OBJECT)/ParaPFor.o -o $(EXECUTABLE)

$(OBJECT)/main.o: main.cpp $(PARAPFOR)/ParaPFor.h
	$(CC) $(CCFLAGS) -I$(PARAPFOR) -c main.cpp -o $(OBJECT)/main.o

$(OBJECT)/ParaPFor.o: $(PARAPFOR)/ParaPFor.cpp $(PARAPFOR)/ParaPFor.h
	$(CC) $(CCFLAGS) -I$(PARAPFOR) -c $(PARAPFOR)/ParaPFor.cpp -o $(OBJECT)/ParaPFor.o


clean:
	rm -f $(EXECUTABLE) 
	rm -f $(OBJECT)/*
