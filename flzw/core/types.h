#ifndef FLZW_TYPES
#define FLZW_TYPES

#include <stddef.h>
#include <stdint.h>

#define FLZW_TYPE_LIST \
    FLZW_TYPE(8)  \
    FLZW_TYPE(16)    \
    FLZW_TYPE(32) \
    FLZW_TYPE(64)    \

#define FLZW_TYPE(s) \
    typedef uint##s##_t u##s; \
    typedef int##s##_t i##s;
    FLZW_TYPE_LIST
#undef FLZW_TYPE
#undef FLZW_TYPE_LIST

typedef uintptr_t usize;
typedef intptr_t isize;
typedef u8 byte;

#ifdef nil
#   undef nil
#endif

#define nil NULL

#endif