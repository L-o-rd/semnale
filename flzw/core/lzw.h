#ifndef FLZW_CORE
#define FLZW_CORE

#include <core/types.h>

#define FLZW_CLEAR_CODE (256)
#define FLZW_EOI_CODE (257)

typedef struct {
    usize max_symbol;
    usize reset, eoi;
    usize max_bits;
    usize start;
} flzw_config;

typedef struct {
    size_t size, capacity;
    byte * raw;
} ByteStream;

#define flzw_config_default (flzw_config) { \
    .max_symbol = (1ull << 16), \
    .max_bits = 16, \
    .start = 257, \
    .reset = 1, \
    .eoi = 0, \
}

#define flzw_config_gif (flzw_config) { \
    .max_symbol = (1ull << 16), \
    .max_bits = 16, \
    .start = 258, \
    .reset = 1, \
    .eoi = 1, \
}

int flzw_decode(const byte *, const size_t, const flzw_config, ByteStream *);
int flzw_compress(const char *, const char *, const flzw_config);
int flzw_encode(const byte *, const size_t, const flzw_config);
int flzw_decompress(const char *, const char *);
void flzw_break(int);
void flzw_make(int);

#endif