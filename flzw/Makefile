CFLAGS := -Wall -Wextra -pedantic -std=c99 \
			-Wno-newline-eof -I.
CC := clang
LD := ld

SOURCE := $(wildcard ./*.c) \
	   		$(wildcard ./core/*.c) \
			$(wildcard ./trie/*.c) \
			$(wildcard ./suite/*.c)

HEADERS := $(wildcard *.h) \
	   		$(wildcard ./core/*.h) \
	   		$(wildcard ./trie/*.h) \
	   		$(wildcard ./suite/*.h)

BUILD_DIR := build
DEBUG_DIR := $(BUILD_DIR)/debug
RELEASE_DIR := $(BUILD_DIR)/release

DEBUG_FLAGS := -g -ggdb -gdwarf-2 -Og -DFLZW_DEBUG -fsanitize=address
RELEASE_FLAGS := -O3 -DFLZW_RELEASE -flto -march=native -mtune=native

RELEASE_TARGET := $(RELEASE_DIR)/flzw
DEBUG_TARGET := $(DEBUG_DIR)/flzw

RELEASE_OBJS := $(patsubst %.c, $(RELEASE_DIR)/%.o, $(SOURCE))
DEBUG_OBJS := $(patsubst %.c, $(DEBUG_DIR)/%.o, $(SOURCE))

.PHONY: all debug release clean run tests

all: debug release
debug: $(DEBUG_TARGET)
release: $(RELEASE_TARGET)

$(DEBUG_TARGET): $(DEBUG_OBJS)
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) -o $@ $^

$(RELEASE_TARGET): $(RELEASE_OBJS)
	$(CC) $(CFLAGS) $(RELEASE_FLAGS) -o $@ $^

$(DEBUG_DIR)/%.o: %.c $(HEADERS) | $(DEBUG_DIR)
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) -c -o $@ $<

$(RELEASE_DIR)/%.o: %.c $(HEADERS) | $(RELEASE_DIR)
	$(CC) $(CFLAGS) $(RELEASE_FLAGS) -c -o $@ $<

$(DEBUG_DIR) $(RELEASE_DIR):
	@mkdir -p $@
	@mkdir -p $@/core
	@mkdir -p $@/trie
	@mkdir -p $@/suite

enable-tests:
	$(eval CFLAGS += -DFLZW_TESTS)

tests: enable-tests debug
	./$(DEBUG_TARGET)

run: release
	./$(RELEASE_TARGET)

clean:
	@rm -rf $(BUILD_DIR)
