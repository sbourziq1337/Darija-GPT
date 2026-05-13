CXX      := g++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -Wpedantic
CPPFLAGS := -Isrc

SRC_DIR  := src

# Common source files (exclude programs with their own main())
COMMON_SRCS := $(filter-out $(SRC_DIR)/test_main.cpp $(SRC_DIR)/chat.cpp $(SRC_DIR)/main.cpp,$(wildcard $(SRC_DIR)/*.cpp))
COMMON_OBJS := $(COMMON_SRCS:.cpp=.o)

TARGET_LLM  := llm
TARGET_CHAT := chat

all: $(TARGET_LLM) $(TARGET_CHAT)

# Main training/generation program
$(TARGET_LLM): $(COMMON_OBJS) $(SRC_DIR)/main.o
	$(CXX) $(CXXFLAGS) -o $@ $^

# Interactive chat program
$(TARGET_CHAT): $(COMMON_OBJS) $(SRC_DIR)/chat.o
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(COMMON_OBJS) $(SRC_DIR)/main.o $(SRC_DIR)/chat.o $(TARGET_LLM) $(TARGET_CHAT)

.PHONY: all clean
