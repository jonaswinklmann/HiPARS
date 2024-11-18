CC=c++
INCLUDE=-I../extern/pybind11/include -Iinclude -I../extern/eigen -I../extern/spdlog/include $(shell python3-config --includes)
CXXFLAGS=-O3 -Wall -shared -std=c++17 -fPIC $(INCLUDE)

TARGET_DIR=../
SRC_DIR	:= src
OBJ_DIR	:= obj
BIN_DIR	:= lib

SOURCES := $(wildcard $(SRC_DIR)/*.cpp)

OBJECTS	:= $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

all: module
fresh:
	-rm $(OBJECTS)
	-rm $(BIN_DIR)/resorting_cpp$(shell python3-config --extension-suffix)
	make module

module: 
	make $(BIN_DIR)/resorting_cpp
$(BIN_DIR)/resorting_cpp: $(OBJECTS) | $(BIN_DIR) $(TARGET_DIR)
	$(CC) -o $@$(shell python3-config --extension-suffix) $^ $(CXXFLAGS)
	cp $@$(shell python3-config --extension-suffix) $(TARGET_DIR)$(notdir $@$(shell python3-config --extension-suffix))

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CC) -c $< -o $@ $(CXXFLAGS)
$(BIN_DIR) $(OBJ_DIR) $(TARGET_DIR):
	mkdir -p $@