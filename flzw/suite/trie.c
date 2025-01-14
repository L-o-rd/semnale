#include <suite/test.h>
#include <trie/node.h>
#include <stdio.h>

/**
    Trie Show tests.
 */

int trie_show_nil(void) {
    trie_show(nil);
    return flzw_assert(1, "trie_show(nil)");
}

int trie_show_empty(void) {
    TrieNode * root = trie_make();
    trie_show(root);
    trie_break(&root);
    return flzw_assert(1, "trie_show(root) (empty)");
}

/**
    Trie Look tests.
 */

int trie_look_nil(void) {
    return flzw_assert(!trie_look(nil, (const byte *) "abcd", 4, nil), "trie_look(nil, \"abcd\", 4, nil)");
}

int trie_look_none(void) {
    TrieNode * root = trie_make();
    int result = flzw_assert(!trie_look(root, (const byte *) "abcd", 4, nil), "trie_look(root, \"abcd\", 4, nil) (empty)");
    trie_break(&root);
    return result;
}

int trie_look_valid(void) {
    TrieNode * root = trie_make();
    trie_add(root, (const byte *) "abcd", 4, 144);
    int result = flzw_assert(trie_look(root, (const byte *) "abcd", 4, nil), "trie_look(root, \"abcd\", 4, nil) (entry)");
    trie_break(&root);
    return result;
}

/**
    Trie Add tests.
 */

int trie_add_short(void) {
    TrieNode * root = trie_make();
    usize f0, f1, f2, result = 1;

    trie_add(root, (const byte *) "abcd", 4, 255);
    trie_add(root, (const byte *)   "ab", 2, 256);
    trie_add(root, (const byte *) "bcde", 4, 257);
    trie_show(root);

    result = result && trie_look(root, (const byte *) "abcd", 4, &f0) &&
        trie_look(root, (const byte *)   "ab", 2, &f1) &&
        trie_look(root, (const byte *) "bcde", 4, &f2);

    trie_break(&root);
    result = result && flzw_assert(
        (f0 == 255) && (f1 == 256) && (f2 == 257),
        "trie has correct items: (\"abcd\", 255), (\"ab\", 256), (\"bcde\", 257)"
    );

    return result;
}

int trie_add_long(void) {
    TrieNode * root = trie_make();
    usize f0, f1, result = 1;

    trie_add(root, (const byte *) "abcdefghijklmnopqrstuvwxyz", 26, 255);
    trie_add(root, (const byte *) "abcdefghtuvwxyz", 15, 256);
    trie_add(root, (const byte *) "abcdefghijklmnopqrstuvwxyz", 26, 258);
    trie_show(root);

    result = result && trie_look(root, (const byte *) "abcdefghijklmnopqrstuvwxyz", 26, &f0) &&
        trie_look(root, (const byte *) "abcdefghtuvwxyz", 15, &f1);

    trie_break(&root);
    result = result && flzw_assert(
        (f0 == 258) && (f1 == 256),
        "trie has correct items."
    );

    return result;
}

/**
    Trie Break tests.
 */

int trie_break_valid(void) {
    TrieNode * root = trie_make();
    trie_break(&root);
    return flzw_assert(root == nil, "trie_break(root) (valid)");
}

int trie_break_nil(void) {
    trie_break(nil);
    return flzw_assert(1, "trie_break(nil)");
}

int trie_tests(void) {
    printf("\n~ Trie Tests:\n");

    probe_t probes[] = {
        trie_break_nil,
        trie_break_valid,
        trie_show_nil,
        trie_show_empty,
        trie_look_nil,
        trie_look_none,
        trie_look_valid,
        trie_add_short,
        trie_add_long,
        nil
    };

    return flzw_tests(probes);
}