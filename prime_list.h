#pragma once
#include <stdint.h>

typedef struct {
    uint32_t *primes;
    uint32_t  count;
} prime_list_t;

#ifdef __cplusplus
extern "C" {
#endif

/* Load binary prime file: [uint32_t count][uint32_t p0][uint32_t p1]... */
int  prime_list_load(prime_list_t *pl, const char *path);
void prime_list_free(prime_list_t *pl);

#ifdef __cplusplus
}
#endif
