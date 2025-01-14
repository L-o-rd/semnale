#include <core/stream.h>
#include <stdlib.h>
#include <stdio.h>

static inline __attribute__ ((always_inline)) byte brev(byte n) {
    n = (n & 0xF0) >> 4 | (n & 0x0F) << 4;
    n = (n & 0xCC) >> 2 | (n & 0x33) << 2;
    n = (n & 0xAA) >> 1 | (n & 0x55) << 1;
    return n;
}

void stream_push(Stream * stream, usize value, int bits) {
    while (bits > 0) {
        stream->acc |= (value & 1) << (stream->bits);
        --stream->bits;
        value >>= 1;
        --bits;

        if (!((stream->bits + 1) & 7)) {
            stream->bytes[stream->size++] = brev((byte) (stream->acc >> 56));
            if (stream->size >= stream->capacity) {
                stream->capacity <<= 1;
                stream->bytes = realloc(stream->bytes, stream->capacity * sizeof(byte));
            }

            stream->bits = 63;
            stream->acc = 0;
        }
    }
}

void stream_flush(Stream * stream) {
    if (stream->bits >= 63) return;

    stream->bytes[stream->size++] = brev((byte) (stream->acc >> 56));
    if (stream->size >= stream->capacity) {
        stream->capacity <<= 1;
        stream->bytes = realloc(stream->bytes, stream->capacity * sizeof(byte));
    }
    
    stream->bits = 63;
    stream->acc = 0;
}

void stream_break(Stream ** ps) {
    if (ps == nil) return;
    if (*ps == nil) return;
    register Stream * stream = *ps;
    free(stream->bytes);
    stream->size = 0;
    free(stream);
    *ps = nil;
}

Stream * stream_make(void) {
    Stream * stream = malloc(sizeof *stream);
    stream->capacity = 128;
    stream->acc = stream->size = 0;
    stream->bits = (sizeof(stream->acc) << 3) - 1;
    stream->bytes = malloc(sizeof(byte) * stream->capacity);
    return stream;
}