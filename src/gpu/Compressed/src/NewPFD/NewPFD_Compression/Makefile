CC = g++
CCFLAGS = -std=c++0x -O3
PARAPFOR = ../../../include/
OBJECT = object
EXECUTABLE = NewPFD_Compression

$(EXECUTABLE): $(OBJECT)/main.o $(OBJECT)/NewPFor.o
	$(CC) $(CCFLAGS) -I$(PARAPFOR) $(OBJECT)/main.o $(OBJECT)/NewPFor.o -o $(EXECUTABLE)

$(OBJECT)/main.o: main.cpp $(PARAPFOR)/NewPFor.h
	$(CC) $(CCFLAGS) -I$(PARAPFOR) -c main.cpp -o $(OBJECT)/main.o

$(OBJECT)/NewPFor.o: $(PARAPFOR)/NewPFor.cpp $(PARAPFOR)/NewPFor.h
	$(CC) $(CCFLAGS) -I$(PARAPFOR) -c $(PARAPFOR)/NewPFor.cpp -o $(OBJECT)/NewPFor.o


clean:
	rm -f $(EXECUTABLE) 
	rm -f $(OBJECT)/*
