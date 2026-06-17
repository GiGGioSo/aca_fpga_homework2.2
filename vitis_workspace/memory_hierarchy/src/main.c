#include <xil_printf.h>  // Lightweight version of printf
#include <xparameters.h> // Defines all the components in the memory map
#include <xio.h>         // Required for reading / writing memory mapped registers

#include <stdlib.h> // Required for malloc, srand/rand ..
#include <stdio.h>

#include "matmul.h"
#include "timer.h"

#define ROWS_A 128
#define COLS_A 128
#define ROWS_B 128
#define COLS_B 128

#define VECTOR_A 4096 * 256

void fill_vector_random(int *matrix, int size)
{
    for (int j = 0; j < size; j++)
    {
        matrix[j] = rand() % 4096; // Random double between 0 and 4095
    }
}

void fill_indirection_vector_random(int *matrix, int size)
{
    for (int j = 0; j < size; j++)
    {
        matrix[j] = rand() % size; // Random int between 0 and size to creat an indirection Vector
    }
}

void kernel1(int *A, int size, int offset)
{
    int i;
    for (i = 0; i < size - offset; i++)
        A[i] += A[i + offset];
    return;
}

void kernel1Optimisation(int *A, int size, int offset)
{
    int i;
    // Loop unrolling
    for (i = 0; i < size - offset - (size - offset) % 4; i += 4)
    {
        A[i] += A[i + offset];
        A[i + 1] += A[i + offset + 1];
        A[i + 2] += A[i + offset + 2];
        A[i + 3] += A[i + offset + 3];
    }
    // Handeling the edgecase when size is not a multiple of 5
    i++;
    if (i >= size - offset)
        return;
    A[i] += A[i + offset];
    i++;
    if (i >= size - offset)
        return;
    A[i] += A[i + offset];
    i++;
    if (i >= size - offset)
        return;
    A[i] += A[i + offset];
    return;
}

void kernel2(int *A, int size)
{
    int i;
    for (i = 3; i < size; i++)
        A[i] = A[i - 1] + A[i - 2] * A[i - 3];
}

void kernel2Optimisation(int *A, int size)
{
    int i;
    for (i = 3; i < size - size % 4; i += 4)
    {
        A[i] = A[i - 1] + A[i - 2] * A[i - 3];
        A[i + 1] = A[i] + A[i - 1] * A[i - 2];
        A[i + 2] = A[i + 1] + A[i] * A[i - 1];
        A[i + 3] = A[i + 2] + A[i + 1] * A[i];
    }
    i++;
    if (i >= size)
        return;
    A[i] = A[i - 1] + A[i - 2] * A[i - 3];
    i++;
    if (i >= size)
        return;
    A[i] = A[i - 1] + A[i - 2] * A[i - 3];
    i++;
    if (i >= size)
        return;
    A[i] = A[i - 1] + A[i - 2] * A[i - 3];
}

void kernel3(float *h, float *w, int *idx, int size)
{
    for (int i = 0; i < size; ++i)
        h[idx[i]] = h[idx[i]] + w[i];
}

void kernel3Optimisation(float *h, float *w, int *idx, int size)
{
    int i;
    for (i = 0; i < size - size % 4; i += 4)
    {
        h[idx[i]] = h[idx[i]] + w[i];
        h[idx[i + 1]] = h[idx[i + 1]] + w[i + 1];
        h[idx[i + 2]] = h[idx[i + 2]] + w[i + 2];
        h[idx[i + 3]] = h[idx[i + 3]] + w[i + 3];
    }
    i++;
    if (i >= size)
        return;
    h[idx[i]] = h[idx[i]] + w[i];

    i++;
    if (i >= size)
        return;
    h[idx[i]] = h[idx[i]] + w[i];

    i++;
    if (i >= size)
        return;
    h[idx[i]] = h[idx[i]] + w[i];
}

float kernel4(float *A, float *B, int size)
{
    float sum = 0;
    for (int i = 0; i < size; i++)
    {
        float diff = A[i] - B[i];
        if (diff > 0)
            sum = (sum + diff);
    }
    return sum;
}

float kernel4Optimisation(float *A, float *B, int size)
{
    float sum = 0;
    float diff = 0;
    int i;
    for (i = 0; i < size - size % 4; i += 4)
    {
        // removing the if statment
        diff = A[i] - B[i];
        sum = (sum + diff * (diff > 0));
        diff = A[i + 1] - B[i + 1];
        sum = (sum + diff * (diff > 0));
        diff = A[i + 2] - B[i + 2];
        sum = (sum + diff * (diff > 0));
        diff = A[i + 3] - B[i + 3];
    }
    i++;
    if (i >= size)
        return sum;
    diff = A[i] - B[i];
    sum = (sum + diff * (diff > 0));

    i++;
    if (i >= size)
        return sum;
    diff = A[i] - B[i];
    sum = (sum + diff * (diff > 0));

    i++;
    if (i >= size)
        return sum;
    diff = A[i] - B[i];
    sum = (sum + diff * (diff > 0));

    return sum;
}

int main()
{
    init_platform();
    srand(42);

    int *A = (int *)malloc(VECTOR_A * sizeof(int));
    int *B = (int *)malloc(VECTOR_A * sizeof(int));

    fill_vector_random(A, VECTOR_A);
    fill_vector_random(B, VECTOR_A);

    // Perform matrix multiplication
    start_timer();
    kernel2(A, VECTOR_A);
    double t = stop_timer();

    char s[128] = {};
    printf("Base Time: %.4fs\r\n", t);
    // sprintf(s, "Base Time: %.4fs\r\n", t);
    // xil_printf("%s", s);

    fill_vector_random(A, VECTOR_A);
    fill_vector_random(B, VECTOR_A);

    // Perform matrix multiplication
    start_timer();
    kernel2Optimisation(A, VECTOR_A);
    t = stop_timer();

    printf("Base Time: %.4fs\r\n", t);
    // sprintf(s, "Improvment Time: %.4fs\r\n", t);
    // xil_printf("%s", s);

    free(A);
    cleanup_platform();
    return 0;
}
