#include <suite/test.h>
#include <core/types.h>
#include <stdio.h>

#define COLOR_NORMAL "\x1b[1;37m"
#define COLOR_GREEN "\x1b[1;32m"
#define COLOR_RED "\x1b[1;31m"

int flzw_assert(const int value, const char * message) {
    if (value) printf("`%s` .. %spassed%s.\n", message, COLOR_GREEN, COLOR_NORMAL);
    else printf("`%s` .. %sfailed%s.\n", message, COLOR_RED, COLOR_NORMAL);
    return !!value;
}

int flzw_tests(probe_t probes[]) {
    register probe_t * probe = &probes[0];
    int success = 0, total = 0;

    for ((void) probe; *probe != nil; ++probe) {
        success += (*probe)();
        ++total;
    }

    printf("Tests passed: %d/%d.\n", success, total);
    return success == total;
}

#undef COLOR_NORMAL
#undef COLOR_GREEN
#undef COLOR_RED