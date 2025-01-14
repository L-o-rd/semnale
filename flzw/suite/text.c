#include <suite/test.h>
#include <core/types.h>
#include <sys/wait.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>

const char * files[] = {
    "./suite/text/test.txt.1",
    "./suite/text/test.txt.2",
    "./suite/text/test.txt.4",
    "./suite/text/test.txt.8",
    "./suite/text/test.txt.16",
    nil
};

int text_tests(void) {
    int result = 1;
    printf("\n ~ Text Tests:\n");
    char command[512] = {0}, assertm[512] = {0};
    for (const char ** file = &files[0]; *file != nil; ++file) {
        if (!fork()) {
            char * args[] = {"cp", (char *) *file, "./suite/text/test.txt", nil};
            execvp("cp", args);
            exit(0);
        }

        wait(0);
        if (!fork()) {
            char * args[] = {"./build/release/flzw", "./suite/text/test.txt", "--out", "./suite/text/test.txt.Z1", nil};
            execvp("./build/release/flzw", args);
            exit(0);
        }

        wait(0);
        if (!fork()) {
            char * args[] = {"compress", "./suite/text/test.txt", nil};
            execvp("compress", args);
            exit(0);
        }

        wait(0);
        snprintf(command, sizeof(command), "diff -s ./suite/text/test.txt.Z1 ./suite/text/test.txt.Z");
        snprintf(assertm, sizeof(assertm), "compress(%s) == flzw(%s)", *file, *file);
        result = result && flzw_assert(system(command) == 0, assertm);
        system("rm ./suite/text/test.txt.Z1 ./suite/text/test.txt.Z");
    }

    return result;
}