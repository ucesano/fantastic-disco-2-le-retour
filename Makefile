CC=cc -O2 -Wall -Wextra
CXX=c++ -std=c++17 -O2 -Wall -Wextra
NVCC=nvcc

LIBS=
INCLUDES=-I../../
LIB_FLAGS=-lm


BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src
BATCH_OUT_FOLDER := outputs


MAIN_NAME=main
MAIN_BIN=spmv
MAIN_SRC=$(MAIN_NAME).cu

OBJECTS = $(OBJ_FOLDER)/mmio.o $(OBJ_FOLDER)/mmfmt.o $(OBJ_FOLDER)/mt19937ar.o $(OBJ_FOLDER)/bench.o $(OBJ_FOLDER)/spmv_cpu.o $(OBJ_FOLDER)/spmv_gpu.o

all: $(BIN_FOLDER)/$(MAIN_BIN)

$(OBJ_FOLDER)/mmio.o: $(SRC_FOLDER)/mmio.c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) -c $(SRC_FOLDER)/mmio.c -o $@ $(LIB_FLAGS)

$(OBJ_FOLDER)/mmfmt.o: $(SRC_FOLDER)/mmfmt.c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) -c $(SRC_FOLDER)/mmfmt.c -o $@ $(LIB_FLAGS)

$(OBJ_FOLDER)/mt19937ar.o: $(SRC_FOLDER)/mt19937ar.c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) -c $(SRC_FOLDER)/mt19937ar.c -o $@ $(LIB_FLAGS)

$(OBJ_FOLDER)/bench.o: $(SRC_FOLDER)/bench.cu
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(NVCC) -c $(SRC_FOLDER)/bench.cu -o $@ $(LIB_FLAGS)

$(OBJ_FOLDER)/spmv_cpu.o: $(SRC_FOLDER)/spmv_cpu.c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) -c $(SRC_FOLDER)/spmv_cpu.c -o $@ $(LIB_FLAGS)

$(OBJ_FOLDER)/spmv_gpu.o: $(SRC_FOLDER)/spmv_gpu.cu
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(NVCC) -c $(SRC_FOLDER)/spmv_gpu.cu -o $@ $(LIB_FLAGS)

$(BIN_FOLDER)/$(MAIN_BIN): $(MAIN_SRC) $(OBJECTS)
	mkdir -p $(BIN_FOLDER)
	$(NVCC) $^ -o $@ $(LIBS) $(INCLUDES) $(LIB_FLAGS)

clean:
	rm -rf $(BIN_FOLDER) $(OBJ_FOLDER)
	mv -f logs/test-*.??? .trash/
