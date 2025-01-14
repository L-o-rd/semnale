#include <core/stream.h>
#include <trie/node.h>
#include <core/lzw.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define MAX_WORKING (512)

typedef struct {
    byte value[MAX_WORKING];
    usize size;
    int valid;
} Sequence;

typedef struct {
    size_t capacity, size;
    Sequence * raw;
} Sequences;

static void seqs_add(Sequences * seqs, const byte * value, size_t length, size_t index) {
#ifdef FLZW_DEBUG
    if (index >= seqs->size) {
        if (index > seqs->size) {
            fprintf(stderr, "warn: sequences is way too small?\n");
        }

        Sequence * seq = &seqs->raw[seqs->size++];
        memcpy(seq->value, value, length);
        seq->size = length;
        seq->valid = 1;

        if (seqs->size >= seqs->capacity) {
            seqs->capacity <<= 1;
            seqs->raw = realloc(seqs->raw, sizeof(Sequence) * seqs->capacity);
        }
    } else {
        if (seqs->raw[index].valid) {
            fprintf(stderr, "warn: replacing valid sequence?\n");
        }

        Sequence * seq = &seqs->raw[index];
        memcpy(seq->value, value, length);
        seq->size = length;
        seq->valid = 1;
    }
#else
    (void) index;

    Sequence * seq = &seqs->raw[seqs->size++];
    memcpy(seq->value, value, length);
    seq->size = length;
    seq->valid = 1;

    if (seqs->size >= seqs->capacity) {
        seqs->capacity <<= 1;
        seqs->raw = realloc(seqs->raw, sizeof(Sequence) * seqs->capacity);
    }
#endif
}

Stream * bitstream = nil;
TrieNode * table = nil;
Sequences * seqs = nil;

void flzw_make(int search) {
    table = trie_make();
    for (usize i = 0; i < 256; ++i) 
        trie_add(table, (const byte *) &i, 1, i);

    if (search) {
        seqs = malloc(sizeof *seqs);
        seqs->capacity = 512;
        seqs->raw = malloc(sizeof(Sequence) * seqs->capacity);
        memset(seqs->raw, 0, sizeof(Sequence) * seqs->capacity);
        seqs->size = 0;

        for (usize i = 0; i < 256; ++i)
            seqs_add(seqs, (const byte *) &i, 1, i);
        
        usize clear = FLZW_CLEAR_CODE;
        seqs_add(seqs, (const byte *) &clear, 2, clear);
    } else {
        bitstream = stream_make();
    }
}

void flzw_break(int search) {
    trie_break(&table);

    if (search) {
        free(seqs->raw);
        seqs->size = 0;
        free(seqs);
        seqs = nil;
    } else {
        stream_break(&bitstream);
    }
}

int flzw_encode(const byte * stream, const size_t length, const flzw_config config) {
    byte working[MAX_WORKING] = {0}, augmented[MAX_WORKING] = {0};
    usize bits = ((sizeof(byte) << 3) + 1), wlen = 0, alen = 0;

    usize result = 1, symbol;
    usize nextsym = config.start;
    register const byte * now = stream;
    for ((void) now; (now - stream) < (isize) length; ++now) {
        memcpy((void *) augmented, (const void *) working, wlen);
        alen = wlen, augmented[alen++] = *now;

        if (trie_look(table, (const byte *) &augmented[0], alen, nil)) {
            memcpy((void *) working, (const void *) augmented, alen), wlen = alen;
        } else if (nextsym >= config.max_symbol) {
#ifdef FLZW_DEBUG
            if (!trie_look(table, (const byte *) &working[0], wlen, &symbol)) {
                printf("fatal: table is full, working should be correct.\n"), result = 0;
                goto failed;
            }
#else
            (void) trie_look(table, (const byte *) &working[0], wlen, &symbol);
#endif

            stream_push(bitstream, symbol, bits);
            wlen = 0, working[wlen++] = *now;
        } else {
            trie_add(table, (const byte *) &augmented[0], alen, nextsym++);

#ifdef FLZW_DEBUG
            if (!trie_look(table, (const byte *) &working[0], wlen, &symbol)) {
                printf("fatal: working was created correctly, should not be here.\n"), result = 0;
                goto failed;
            }
#else
            (void) trie_look(table, (const byte *) &working[0], wlen, &symbol);
#endif

            stream_push(bitstream, symbol, bits);
            wlen = 0, working[wlen++] = *now;
            if (nextsym > (1ull << bits)) ++bits;
        }
    }

    if (wlen > 0) {
#ifdef FLZW_DEBUG
        if (!trie_look(table, (const byte *) &working[0], wlen, &symbol)) {
            printf("fatal: last symbol should be available.\n"), result = 0;
            goto failed;
        }
#else
        (void) trie_look(table, (const byte *) &working[0], wlen, &symbol);
#endif

        stream_push(bitstream, symbol, bits);
    }

    stream_flush(bitstream);
#ifdef FLZW_DEBUG
failed:;
#endif
    return result;
}

static inline __attribute__ ((always_inline)) void bstream_push(ByteStream * stream, const byte b) {
    stream->raw[stream->size++] = b;
    if (stream->size >= stream->capacity) {
        stream->capacity <<= 1;
        stream->raw = realloc(stream->raw, sizeof(byte) * stream->capacity);
    }
}

int flzw_decode(const byte * stream, const size_t length, const flzw_config config, ByteStream * ostream) {
    byte working[MAX_WORKING] = {0}, augmented[MAX_WORKING] = {0};
    usize bits = ((sizeof(byte) << 3) + 1), wlen = 0, alen = 0;
    const usize total = length << 3;
    Sequence seq = {0};

    register usize now = 0;
    register usize seek = 0, mod = 0;
    usize nextsym = config.start, symbol = 0;
    usize acc = 0, mask = (1ull << bits) - 1;
    while (seek + 4 < length) {
        seek = now >> 3, mod = now & 7;
        acc = *(u32 *) (stream + seek), acc >>= mod;

        symbol = acc & mask;
        if (symbol == FLZW_CLEAR_CODE) {
            fprintf(stderr, "fatal: clear code encountered.\n");
            return 0;
        }

        if (symbol < nextsym) {
            seq = seqs->raw[symbol];
        } else {
            memcpy(seq.value, working, wlen), seq.size = wlen;
            seq.value[seq.size++] = working[0];
        }

        for (usize i = 0; i < seq.size; ++i) {
            bstream_push(ostream, seq.value[i]);
            memcpy(augmented, working, wlen), alen = wlen;
            augmented[alen++] = seq.value[i];
            if (trie_look(table, augmented, alen, nil)) {
                memcpy(working, augmented, alen), wlen = alen;
            } else {
                trie_add(table, augmented, alen, nextsym++);
                seqs_add(seqs, augmented, alen, 
            #ifdef FLZW_DEBUG
                    nextsym - 1
            #else
                    0
            #endif
                );

                if ((nextsym > (1ull << bits)) && (bits < config.max_bits)) {
                    ++bits, mask = (1ull << bits) - 1;
                }

                working[0] = seq.value[i];
                wlen = 1;
            }
        }

        now += bits;
    }

    acc = 0;
    while (now + bits <= total) {
        seek = now >> 3, mod = now & 7;
        acc = *(stream + seek);
        acc |= (seek + 1 < length) ? *(stream + seek + 1) <<  8 : 0x00;
        acc |= (seek + 2 < length) ? *(stream + seek + 2) << 16 : 0x00;
        acc |= (seek + 3 < length) ? *(stream + seek + 3) << 24 : 0x00;
        acc >>= mod;

        symbol = acc & mask;
        if (symbol == FLZW_CLEAR_CODE) {
            fprintf(stderr, "fatal: clear code encountered.\n");
            return 0;
        }

        if (symbol < nextsym) {
            seq = seqs->raw[symbol];
        } else {
            memcpy(seq.value, working, wlen), seq.size = wlen;
            seq.value[seq.size++] = working[0];
        }

        for (usize i = 0; i < seq.size; ++i) {
            bstream_push(ostream, seq.value[i]);
            memcpy(augmented, working, wlen), alen = wlen;
            augmented[alen++] = seq.value[i];
            if (trie_look(table, augmented, alen, nil)) {
                memcpy(working, augmented, alen), wlen = alen;
            } else {
                trie_add(table, augmented, alen, nextsym++);
                seqs_add(seqs, augmented, alen, 
            #ifdef FLZW_DEBUG
                    nextsym - 1
            #else
                    0
            #endif
                );

                if ((nextsym > (1ull << bits)) && (bits < config.max_bits)) {
                    ++bits, mask = (1ull << bits) - 1;
                }

                working[0] = seq.value[i];
                wlen = 1;
            }
        }

        now += bits;
    }

    return 1;
}

#define CHUNK_SIZE (512ull << 10)

static int flzw_encode_chunked(FILE * source, const flzw_config config) {
    byte working[MAX_WORKING] = {0}, augmented[MAX_WORKING] = {0};
    usize bits = ((sizeof(byte) << 3) + 1), wlen = 0, alen = 0;

    usize ssize = 0;
    byte * stream = nil;
    usize result = 1, symbol;
    usize nextsym = config.start;
    register const byte * now = stream;
    stream = malloc(sizeof(byte) * CHUNK_SIZE);
    while ((ssize = fread(stream, sizeof *stream, CHUNK_SIZE, source)) > 0) {
        for (now = stream; (now - stream) < (isize) ssize; ++now) {
            memcpy((void *) augmented, (const void *) working, wlen);
            alen = wlen, augmented[alen++] = *now;

            if (trie_look(table, (const byte *) &augmented[0], alen, nil)) {
                memcpy((void *) working, (const void *) augmented, alen), wlen = alen;
            } else if (nextsym >= config.max_symbol) {
    #ifdef FLZW_DEBUG
                if (!trie_look(table, (const byte *) &working[0], wlen, &symbol)) {
                    printf("fatal: table is full, working should be correct.\n"), result = 0;
                    goto failed;
                }
    #else
                (void) trie_look(table, (const byte *) &working[0], wlen, &symbol);
    #endif

                stream_push(bitstream, symbol, bits);
                wlen = 0, working[wlen++] = *now;
            } else {
                trie_add(table, (const byte *) &augmented[0], alen, nextsym++);

    #ifdef FLZW_DEBUG
                if (!trie_look(table, (const byte *) &working[0], wlen, &symbol)) {
                    printf("fatal: working was created correctly, should not be here.\n"), result = 0;
                    goto failed;
                }
    #else
                (void) trie_look(table, (const byte *) &working[0], wlen, &symbol);
    #endif

                stream_push(bitstream, symbol, bits);
                wlen = 0, working[wlen++] = *now;
                if (nextsym > (1ull << bits)) ++bits;
            }
        }
    }

    if (wlen > 0) {
#ifdef FLZW_DEBUG
        if (!trie_look(table, (const byte *) &working[0], wlen, &symbol)) {
            printf("fatal: last symbol should be available.\n"), result = 0;
            goto failed;
        }
#else
        (void) trie_look(table, (const byte *) &working[0], wlen, &symbol);
#endif

        stream_push(bitstream, symbol, bits);
    }

    stream_flush(bitstream);
#ifdef FLZW_DEBUG
failed:;
#endif
    return free(stream), result;
}

static int flzw_decode_chunked(FILE * source, const flzw_config config, ByteStream * ostream) {
    byte working[MAX_WORKING] = {0}, augmented[MAX_WORKING] = {0};
    usize bits = ((sizeof(byte) << 3) + 1), wlen = 0, alen = 0;
    Sequence seq = {0};
    int result = 1;

    register usize now = 0;
    register usize seek = 0, mod = 0;
    usize nextsym = config.start, symbol = 0;
    byte * stream = malloc(sizeof(byte) * CHUNK_SIZE);
    usize acc = 0, lacc, lbts = 0, mask = (1ull << bits) - 1, ssize;
    while ((ssize = fread(stream, sizeof *stream, CHUNK_SIZE, source)) > 0) {
        usize total = ssize << 3;
        now = seek = mod = 0;
        acc = 0;

        if (lbts > 0) {
            acc = (*stream) << lbts;
            acc |= lacc;
            symbol = acc & mask;
            if (symbol == FLZW_CLEAR_CODE) {
                fprintf(stderr, "fatal: clear code encountered.\n");
                result = 0;
                goto failed;
            }

            if (symbol < nextsym) {
                seq = seqs->raw[symbol];
            } else {
                memcpy(seq.value, working, wlen), seq.size = wlen;
                seq.value[seq.size++] = working[0];
            }

            for (usize i = 0; i < seq.size; ++i) {
                bstream_push(ostream, seq.value[i]);
                memcpy(augmented, working, wlen), alen = wlen;
                augmented[alen++] = seq.value[i];
                if (trie_look(table, augmented, alen, nil)) {
                    memcpy(working, augmented, alen), wlen = alen;
                } else {
                    trie_add(table, augmented, alen, nextsym++);
                    seqs_add(seqs, augmented, alen, 
                #ifdef FLZW_DEBUG
                        nextsym - 1
                #else
                        0
                #endif
                    );

                    if ((nextsym > (1ull << bits)) && (bits < config.max_bits)) {
                        ++bits, mask = (1ull << bits) - 1;
                    }

                    working[0] = seq.value[i];
                    wlen = 1;
                }
            }

            now = bits - lbts;
        }

        while (seek + 4 < ssize) {
            seek = now >> 3, mod = now & 7;
            acc = *(u32 *) (stream + seek), acc >>= mod;

            symbol = acc & mask;
            if (symbol == FLZW_CLEAR_CODE) {
                fprintf(stderr, "fatal: clear code encountered.\n");
                result = 0;
                goto failed;
            }

            if (symbol < nextsym) {
                seq = seqs->raw[symbol];
            } else {
                memcpy(seq.value, working, wlen), seq.size = wlen;
                seq.value[seq.size++] = working[0];
            }

            for (usize i = 0; i < seq.size; ++i) {
                bstream_push(ostream, seq.value[i]);
                memcpy(augmented, working, wlen), alen = wlen;
                augmented[alen++] = seq.value[i];
                if (trie_look(table, augmented, alen, nil)) {
                    memcpy(working, augmented, alen), wlen = alen;
                } else {
                    trie_add(table, augmented, alen, nextsym++);
                    seqs_add(seqs, augmented, alen, 
                #ifdef FLZW_DEBUG
                        nextsym - 1
                #else
                        0
                #endif
                    );

                    if ((nextsym > (1ull << bits)) && (bits < config.max_bits)) {
                        ++bits, mask = (1ull << bits) - 1;
                    }

                    working[0] = seq.value[i];
                    wlen = 1;
                }
            }

            now += bits;
        }

        acc = 0;
        while (now + bits <= total) {
            seek = now >> 3, mod = now & 7;
            acc = *(stream + seek);
            acc |= (seek + 1 < ssize) ? *(stream + seek + 1) <<  8 : 0x00;
            acc |= (seek + 2 < ssize) ? *(stream + seek + 2) << 16 : 0x00;
            acc |= (seek + 3 < ssize) ? *(stream + seek + 3) << 24 : 0x00;
            acc >>= mod;

            symbol = acc & mask;
            if (symbol == FLZW_CLEAR_CODE) {
                result = 0;
                goto failed;
            }

            if (symbol < nextsym) {
                seq = seqs->raw[symbol];
            } else {
                memcpy(seq.value, working, wlen), seq.size = wlen;
                seq.value[seq.size++] = working[0];
            }

            for (usize i = 0; i < seq.size; ++i) {
                bstream_push(ostream, seq.value[i]);
                memcpy(augmented, working, wlen), alen = wlen;
                augmented[alen++] = seq.value[i];
                if (trie_look(table, augmented, alen, nil)) {
                    memcpy(working, augmented, alen), wlen = alen;
                } else {
                    trie_add(table, augmented, alen, nextsym++);
                    seqs_add(seqs, augmented, alen, 
                #ifdef FLZW_DEBUG
                        nextsym - 1
                #else
                        0
                #endif
                    );

                    if ((nextsym > (1ull << bits)) && (bits < config.max_bits)) {
                        ++bits, mask = (1ull << bits) - 1;
                    }

                    working[0] = seq.value[i];
                    wlen = 1;
                }
            }

            now += bits;
        }

        mod = now & 7;
        seek = now >> 3;
        lbts = total - now;
        if (lbts > 0) {
            lacc = *(stream + seek);
            lacc |= (lbts > 8) ? *(stream + seek + 1) << 8 : 0x00;
            lacc |= (lbts > 16) ? *(stream + seek + 2) << 16 : 0x00;
            lacc |= (lbts > 24) ? *(stream + seek + 3) << 24 : 0x00;
            lacc >>= mod;
        }
    }

failed:;
    return free(stream), result;
}

#undef MAX_WORKING

int flzw_compress(const char * source, const char * output, const flzw_config config) {
    FILE * todo = fopen(source, "rb");
    FILE * towr = nil;
    int result = 1;

    if (todo == nil) return 0;
    if ((config.max_bits < 9) || (config.max_bits >= 32)) {
        result = 0;
        goto failed;
    }

    flzw_make(0);
    flzw_encode_chunked(todo, config);
    towr = fopen(output, "wb");
    if (towr == nil) {
        result = 0;
        goto failed;
    }

    byte mode = config.reset ? 0x80 : 0x00;
    mode = mode | (config.max_bits & 0x1f);
    fwrite("\x1f\x9d", sizeof(byte), 2, towr);
    fwrite(&mode, sizeof(byte), 1, towr);
    fwrite(bitstream->bytes, sizeof(byte), bitstream->size, towr);
    fflush(towr);

failed:;
    fclose(todo);
    fclose(towr);
    flzw_break(0);
    return result;
}

int flzw_decompress(const char * source, const char * output) {
    FILE * todo = fopen(source, "rb"), * towr = nil;
    if (todo == nil) return 0;
    ByteStream ostream;
    int result = 1;

    byte header[3];
    if (fread(header, sizeof *header, sizeof(header), todo) != sizeof(header)) {
        result = 0;
        goto failed;
    }

    if (memcmp(header, (const byte[2]) {0x1F, 0x9D}, 2)) {
        result = 0;
        goto failed;
    }

    flzw_config config = flzw_config_default;
    usize mode = header[2];

    config.eoi = 0;
    config.max_bits = mode & 0x1f;
    config.reset = (mode >> 7) & 1;
    config.max_symbol = (1ull << (config.max_bits));
    if ((config.max_bits < 9) || (config.max_bits >= 32)) {
        result = 0;
        goto failed;
    }

    towr = fopen(output, "wb");
    if (towr == nil) {
        result = 0;
        goto failed;
    }

    flzw_make(1);
    ostream.size = 0;
    ostream.capacity = CHUNK_SIZE;
    ostream.raw = malloc(sizeof(byte) * CHUNK_SIZE);
    flzw_decode_chunked(todo, config, &ostream);
    fwrite(ostream.raw, sizeof(byte), ostream.size, towr);
    fflush(towr);
failed:;
    free(ostream.raw);
    fclose(todo);
    fclose(towr);
    flzw_break(1);
    return result;
}

#undef CHUNK_SIZE