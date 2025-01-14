#ifndef FLZW_TEST
#define FLZW_TEST

typedef int (* probe_t)(void);

int flzw_assert(const int, const char *);
int flzw_tests(probe_t[]);

#endif