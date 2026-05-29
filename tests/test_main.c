#include <stdio.h>
#include <stdlib.h>

/* Declared in each test file */
int test_math(void);
int test_sieve(void);
int test_phase2(void);

int main(void)
{
    int fail = 0;
    fail += test_math();
    fail += test_sieve();
    fail += test_phase2();

    if (fail == 0)
        printf("All tests PASSED\n");
    else
        printf("%d test(s) FAILED\n", fail);
    return fail ? 1 : 0;
}
