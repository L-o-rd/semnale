#if defined(FLZW_DEBUG) && defined(FLZW_TESTS)
    #include <suite/all.h>

    int main(void) {
        int result = 1;
        result = result && trie_tests();
        result = result && bits_tests();
        result = result && text_tests();
        return !result;
    }
#else
    #include <core/lzw.h>
    #include <unistd.h>
    #include <string.h>
    #include <stdio.h>

    #define OPTIONS \
        OPTION(help, "h", "help") \
        OPTION(output, "o", "out") \
        OPTION(decompress, "d", "decompress") 

    static flzw_config config = flzw_config_default;
    static const char * source = nil;
    static const char * output = nil;
    static int compress = 1;

    static int opt_help(char *** argv) {
        (void) argv;
        return 0;
    }

    static int opt_output(char *** argv) {
        char ** now = ++(*argv);
        if (*now == nil) {
            fprintf(stderr, "error: no <output> was given..\n");
            return 1;
        }

        output = (const char *) *now, ++now;
        return 0;
    }

    static int opt_decompress(char *** argv) {
        (void) argv;
        return (compress = 0);
    }

    int main(int argc, char ** argv) {
        if (argc < 2) {
            fprintf(stderr, "error: nothing to process..\n");
            return 1;
        }

        for (++argv; *argv != nil; ++argv) {
            #define OPTION(fn, s, l) \
                if (!strcmp("-"s, *argv) || !strcmp("--"l, *argv)) { \
                    if (opt_ ## fn (&argv)) { \
                        return 1; \
                    } \
                    \
                    continue; \
                } \

                OPTIONS
            #undef OPTION

            if (source == nil) {
                source = *argv;
                continue;
            }

            fprintf(stderr, "error: invalid option `%s`..\n", *argv);
            return 1;
        }

        if (source == nil) {
            fprintf(stderr, "error: no <source> was given..\n");
            return 1;
        }

        if (access(source, F_OK)) {
            fprintf(stderr, "error: %s is not valid..\n", source);
            return 1;
        }

        char out[512] = {0};
        if (output == nil) {
            strncpy(out, source, sizeof(out));
            if (compress) strcat(out, ".Z");
            else {
                char * ext = strstr(out, ".Z");
                if (ext != nil) *ext = 0;
                else strcat(out, ".d");
            }

            output = (const char *) &out[0];
        }

        if (compress) return !flzw_compress(source, output, config);
        return !flzw_decompress(source, output);
    }

    #undef OPTIONS
#endif