CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -Iinclude -Imodel

# Directories
SRC_DIR = src
MODEL_DIR = model
INCLUDE_DIR = include
OBJ_DIR = obj

# Source files - all .cpp files are in src/ and model/
SRC_FILES = $(SRC_DIR)/conv2d_relu.cpp \
            $(SRC_DIR)/edgemb1.cpp \
            $(SRC_DIR)/edgemb2.cpp \
            $(SRC_DIR)/specemb.cpp \
            $(SRC_DIR)/attention.cpp \
            $(SRC_DIR)/mixed_freq_attention.cpp \
            $(SRC_DIR)/decoder.cpp \
            $(SRC_DIR)/utils.cpp

MODEL_FILES = $(MODEL_DIR)/weights_edge_embedding_1.cpp \
              $(MODEL_DIR)/weights_edge_embedding_2.cpp \
              $(MODEL_DIR)/weights_specemb.cpp \
              $(MODEL_DIR)/weights_attention.cpp \
              $(MODEL_DIR)/weights_decoder.cpp

# Object files
SRC_OBJECTS = $(SRC_FILES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
MODEL_OBJECTS = $(MODEL_FILES:$(MODEL_DIR)/%.cpp=$(OBJ_DIR)/%.o)
MAIN_OBJECT = $(OBJ_DIR)/main.o

ALL_OBJECTS = $(SRC_OBJECTS) $(MODEL_OBJECTS) $(MAIN_OBJECT)

# Target executable
TARGET = neural_network

# Default target
all: $(TARGET)

# Create obj directory if it doesn't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Link object files to create executable
$(TARGET): $(OBJ_DIR) $(ALL_OBJECTS)
	$(CXX) $(ALL_OBJECTS) -o $(TARGET)

# Compile main.cpp
$(OBJ_DIR)/main.o: main.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c main.cpp -o $(OBJ_DIR)/main.o

# Compile src files to object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile model files to object files
$(OBJ_DIR)/%.o: $(MODEL_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up generated files
clean:
	rm -rf $(OBJ_DIR) $(TARGET)

# Rebuild everything
rebuild: clean all

# Dependencies
$(OBJ_DIR)/main.o: main.cpp $(INCLUDE_DIR)/conv2d_relu.h $(INCLUDE_DIR)/edgemb1.h $(INCLUDE_DIR)/edgemb2.h $(INCLUDE_DIR)/specemb.h $(INCLUDE_DIR)/mixed_freq_attention.h $(INCLUDE_DIR)/decoder.h $(INCLUDE_DIR)/utils.h $(MODEL_DIR)/weights_edge_embedding_1.h $(MODEL_DIR)/weights_edge_embedding_2.h $(MODEL_DIR)/weights_specemb.h $(MODEL_DIR)/weights_attention.h $(MODEL_DIR)/weights_decoder.h

$(OBJ_DIR)/conv2d_relu.o: $(SRC_DIR)/conv2d_relu.cpp $(INCLUDE_DIR)/conv2d_relu.h
$(OBJ_DIR)/edgemb1.o: $(SRC_DIR)/edgemb1.cpp $(INCLUDE_DIR)/edgemb1.h $(INCLUDE_DIR)/conv2d_relu.h $(MODEL_DIR)/weights_edge_embedding_1.h
$(OBJ_DIR)/edgemb2.o: $(SRC_DIR)/edgemb2.cpp $(INCLUDE_DIR)/edgemb2.h $(INCLUDE_DIR)/conv2d_relu.h $(MODEL_DIR)/weights_edge_embedding_2.h
$(OBJ_DIR)/specemb.o: $(SRC_DIR)/specemb.cpp $(INCLUDE_DIR)/specemb.h $(INCLUDE_DIR)/conv2d_relu.h $(MODEL_DIR)/weights_specemb.h
$(OBJ_DIR)/attention.o: $(SRC_DIR)/attention.cpp $(INCLUDE_DIR)/attention.h $(INCLUDE_DIR)/conv2d_relu.h
$(OBJ_DIR)/mixed_freq_attention.o: $(SRC_DIR)/mixed_freq_attention.cpp $(INCLUDE_DIR)/mixed_freq_attention.h $(INCLUDE_DIR)/attention.h $(MODEL_DIR)/weights_attention.h
$(OBJ_DIR)/decoder.o: $(SRC_DIR)/decoder.cpp $(INCLUDE_DIR)/decoder.h $(INCLUDE_DIR)/conv2d_relu.h $(MODEL_DIR)/weights_decoder.h
$(OBJ_DIR)/utils.o: $(SRC_DIR)/utils.cpp $(INCLUDE_DIR)/utils.h

$(OBJ_DIR)/weights_edge_embedding_1.o: $(MODEL_DIR)/weights_edge_embedding_1.cpp $(MODEL_DIR)/weights_edge_embedding_1.h $(INCLUDE_DIR)/utils.h
$(OBJ_DIR)/weights_edge_embedding_2.o: $(MODEL_DIR)/weights_edge_embedding_2.cpp $(MODEL_DIR)/weights_edge_embedding_2.h $(INCLUDE_DIR)/utils.h
$(OBJ_DIR)/weights_specemb.o: $(MODEL_DIR)/weights_specemb.cpp $(MODEL_DIR)/weights_specemb.h $(INCLUDE_DIR)/utils.h
$(OBJ_DIR)/weights_attention.o: $(MODEL_DIR)/weights_attention.cpp $(MODEL_DIR)/weights_attention.h $(INCLUDE_DIR)/utils.h
$(OBJ_DIR)/weights_decoder.o: $(MODEL_DIR)/weights_decoder.cpp $(MODEL_DIR)/weights_decoder.h $(INCLUDE_DIR)/utils.h

.PHONY: all clean rebuild
