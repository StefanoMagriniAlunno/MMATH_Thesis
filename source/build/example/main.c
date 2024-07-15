#include <stdio.h>
#include "inc/example.h"

int main() {
    int32_t a = 1;
    int32_t b = 2;
    int32_t c = example(a, b);
    printf("The sum of %d and %d is %d\n", a, b, c);
    return 0;
}
