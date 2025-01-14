#ifndef FLZW_TRIE_NODE
#define FLZW_TRIE_NODE

#include <core/types.h>

#define TRIE_CHILDREN (1 << (sizeof(byte) << 3))
#define TRIE_VALID (1)
#define TRIE_NONE (0)

typedef struct TrieNode {
    struct TrieNode * children[TRIE_CHILDREN];
    usize value, valid;
} TrieNode;

int trie_look(const TrieNode *, const byte *, const size_t, usize * const);
void trie_add(TrieNode *, const byte *, const size_t, usize);
void trie_show(const TrieNode *);
void trie_break(TrieNode **);
TrieNode * trie_make(void);

#endif