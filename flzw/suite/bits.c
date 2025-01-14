#include <core/stream.h>
#include <suite/test.h>
#include <stdio.h>

int bits_break_nil(void) {
    stream_break(nil);
    return flzw_assert(1, "stream_break(nil)");
}

int bits_break_valid(void) {
    Stream * bits = stream_make();
    stream_break(&bits);
    return flzw_assert(bits == nil, "stream_break(bits) (valid)");
}

int bits_push_few(void) {
    Stream * bits = stream_make();
    stream_push(bits, 0x123, 8);
    stream_push(bits, 0x018, 4);
    int result = flzw_assert(
        (bits->size == 1) &&
        (bits->bytes[0] == 0x23) &&
        (bits->acc >> 56) == 0x10,
        "bits_push_few (no flush)"
    );

    stream_break(&bits);
    return result;
}

int bits_flush_none(void) {
    Stream * bits = stream_make();
    stream_flush(bits);
    stream_flush(bits);
    stream_flush(bits);
    int result = flzw_assert(
        (bits->size == 0) &&
        (bits->bits == 63) &&
        (bits->acc == 0),
        "bits_flush_none"
    );

    stream_break(&bits);
    return result;
}

int bits_push_many(void) {
    Stream * bits = stream_make();
    stream_push(bits, 0x123, 9);
    stream_push(bits, 0x798, 8);
    stream_flush(bits);          

    int result = flzw_assert(
        (bits->size == 3) &&
        (bits->bytes[0] == 0x23) &&
        (bits->bytes[1] == 0x31) &&
        (bits->bytes[2] == 0x01) &&
        (bits->acc >> 56) == 0x00,
        "bits_push_many (+ flush)"
    );

    stream_break(&bits);
    return result;
}

int bits_tests(void) {
    printf("\n~ Bits Tests:\n");

    probe_t probes[] = {
        bits_break_nil,
        bits_break_valid,
        bits_push_few,
        bits_push_many,
        bits_flush_none,
        nil
    };

    return flzw_tests(probes);
}