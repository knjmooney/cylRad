# compilers and commands
CC        = gcc
CPP       = g++
NVCC      = nvcc
LINKER    = nvcc
DEL_FILE  = rm -f

# linker
CPPFLAGS  = -W -Wall -O2 -std=c++11 -g
NVCCFLAGS = --use_fast_math --std=c++11 -g -G

# directories
BIN       = bin
SRC       = src
IMG       = images

# targets
PROJ      = $(BIN)/cylRad
OBJECTS   = $(BIN)/main.o $(BIN)/radiatorCPU.o $(BIN)/radiatorGPU.o $(BIN)/kernelInteractions.o

# arguments to be passed to a test run
TESTARGS  = -m 64 -n 64 -p 100 -v

$(PROJ): $(OBJECTS)
	$(LINKER) -o $@ $(OBJECTS) $(LDFLAGS)

$(BIN)/radiatorCPU.o: $(BIN)/kernelInteractions.o

$(BIN)/%.o : $(SRC)/%.cu
	$(NVCC) $< -c -o $@ $(NVCCFLAGS) 

$(BIN)/%.o : $(SRC)/%.cpp
	$(CPP) $< -c -o $@ $(CPPFLAGS) 

test: $(PROJ)
	 ./$(PROJ) $(TESTARGS)

# GNUPLOT ANIMATION
# Requires gnuplot 4.6 and ImageMagick 6.7

DATAFILE = $(IMG)/rad400.dat

$(DATAFILE): $(PROJ)
	./$(PROJ) -n 128 -m 128 -p 400 -v -f $(DATAFILE)

animation: $(DATAFILE)
	cd $(IMG); gnuplot plotRad.gp
	convert -coalesce -loop 0 -delay 10 $(IMG)/rad*.png $(IMG)/rad.gif
	rm $(IMG)/rad*.png $(DATAFILE)

.PHONY: clean		
clean:
	$(DEL_FILE) $(PROJ) $(OBJECTS) $(IMG)/*png $(IMG)/*gif $(IMG)/*dat
