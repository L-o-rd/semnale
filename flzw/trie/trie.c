#include <trie/node.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

TrieNode * trie_make(void) {
    TrieNode * root = malloc(sizeof *root);
    memset(root, 0, sizeof *root);
    return root;
}

void trie_add(TrieNode * root, const byte * sequence, const size_t length, usize value) {
    for (size_t i = 0; i < length; ++i) {
        register TrieNode * parent = root;
        register byte aq = *(sequence + i);
        root = root->children[aq];
        if (root == nil) {
            root = malloc(sizeof *root);
            memset(root, 0, sizeof(root->children));
            parent->children[aq] = root;
            root->valid = TRIE_NONE;
        }
    }

    root->valid = TRIE_VALID;
    root->value = value;
}

int trie_look(const TrieNode * root, const byte * sequence, const size_t length, usize * const result) {
    if (root == nil) return 0;
    if (length == 0) return 0;
    if (sequence == nil) return 0;
    for (size_t i = 0; i < length; ++i) {
        register byte aq = *(sequence + i);
        root = root->children[aq];
        if (root == nil) return 0;
    }

    if (result) *result = root->value;
    return 1;
}

#define TRIE_MAX_WORD (256)

static void trie_branch(const TrieNode * root, byte * word, size_t depth) {
    if (root->valid == TRIE_VALID) {
        printf("(");
        for (size_t i = 0; i < depth; ++i) {
            byte v = (0x20 <= word[i]) && (word[i] < 0x80) ? word[i] : '.';
            printf("%c", v);
        }

        printf(", %lu)\n", root->value);
    }

    for (int i = 0; i < TRIE_CHILDREN; ++i) {
        if (root->children[i] != nil) {
            word[depth] = i;
            trie_branch(root->children[i], word, depth + 1);
        }
    }
}

const char * trie_empty = "<trie: empty>";
const char * trie_nil = "<trie: nil>";

void trie_show(const TrieNode * root) {
    byte word[TRIE_MAX_WORD] = {0}, none = 1;
    if (root == nil) {
        printf("%s\n", trie_nil);
        return;
    }

    for (int i = 0; i < TRIE_CHILDREN; ++i) {
        if (root->children[i] != nil) {
            none = 0;
            break;
        }
    }

    if (none) printf("%s\n", trie_empty);
    else trie_branch(root, word, 0);
}

#undef TRIE_MAX_WORD

void trie_break(TrieNode ** root) {
    if (root == nil) return;
    if (*root == nil) return;
    
    for (int i = 0; i < TRIE_CHILDREN; ++i) {
        trie_break(&((*root)->children[i]));
    }

    free(*root), *root = nil;
}