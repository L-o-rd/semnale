#ifndef FLZW_BITSTREAM
#define FLZW_BITSTREAM

#include <core/types.h>

typedef struct {
    int size, capacity;
    byte * bytes;
    usize acc;
    int bits;
} Stream;

void stream_push(Stream *, usize, int);
void stream_break(Stream **);
void stream_flush(Stream *);
Stream * stream_make(void);

#endif