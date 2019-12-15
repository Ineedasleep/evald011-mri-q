CC	= gcc
CPP	= g++
EXE	= mri-q
OBJ	= file.o parboil.o args.o main.o
APP_CXXFLAGS=-O3 -funroll-all-loops -ffast-math
APP_CFLAGS=-O3 -funroll-all-loops -ffast-math
APP_LDFLAGS=-lm -lstdc++
LANGUAGE=c

NVCC        = nvcc
#ARCH=-arch=sm_21
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64

default: $(EXE)

main.o: main.cu computeQ.cu parboil.h
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

parboil.o: parboil.c parboil.h
	$(CC) $(APP_CFLAGS) -c parboil.c

file.o: file.cc file.h
	$(CPP) $(APP_CXXFLAGS) -c file.cc

args.o: args.c 
	$(CC) $(APP_CFLAGS) -c args.c

#computeQ.o: computeQ.cu computeQ.h
	#$(NVCC) -c -o $@ computeQ.cu $(NVCC_FLAGS) 

mri-q: main.o file.o args.o parboil.o
	$(CPP) $(APP_CXXFLAGS) $(OBJ) -o $(EXE) $(APP_LDFLAGS) $(LD_FLAGS)

clean:
	rm -f *.o $(EXE) 
