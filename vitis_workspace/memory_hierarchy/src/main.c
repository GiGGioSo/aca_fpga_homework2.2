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

void kernel1(int *A, int size, int offset)
{
    int i;
    for (i = 0; i < size - offset; i++)
        A[i] += A[i + offset];
}

void kernel1Optimisation(int *A, int size, int offset)
{
    int i;
    // Loop unrolling
    for (i = 0; i < size - offset - (size - offset) % 5; i = i + 5)
    {
        A[i] += A[i + offset];
        A[i + 1] += A[i + offset + 1];
        A[i + 2] += A[i + offset + 2];
        A[i + 3] += A[i + offset + 3];
        A[i + 4] += A[i + offset + 4];
        A[i + 5] += A[i + offset + 5];
    }
    // Handeling the edgecase when size is not a multiple of 5
    i++;
    if (i < size - offset)
        A[i] += A[i + offset];
    i++;
    if (i < size - offset)
        A[i + 1] += A[i + offset + 1];
    i++;
    if (i < size - offset)
        A[i + 2] += A[i + offset + 2];
    i++;
    if (i < size - offset)
        A[i + 3] += A[i + offset + 3];
    i++;
    if (i < size - offset)
        A[i + 4] += A[i + offset + 4];
    i++;
    if (i < size - offset)
        A[i + 5] += A[i + offset + 5];
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
    for (i = 3; i < size - size % 5; i = i + 5)
    {
        A[i] = A[i - 1] + A[i - 2] * A[i - 3];
        A[i + 1] = A[i] + A[i - 1] * A[i - 2];
        A[i + 2] = A[i + 1] + A[i] * A[i - 1];
        A[i + 3] = A[i + 2] + A[i + 1] * A[i];
        A[i + 4] = A[i + 3] + A[i + 2] * A[i + 1];
    }
    i++;
    if (i < size)
        A[i] = A[i - 1] + A[i - 2] * A[i - 3];
    i++;
    if (i < size)
        A[i] = A[i - 1] + A[i - 2] * A[i - 3];
    i++;
    if (i < size)
        A[i] = A[i - 1] + A[i - 2] * A[i - 3];
    i++;
    if (i < size)
        A[i] = A[i - 1] + A[i - 2] * A[i - 3];
    i++;
    if (i < size)
        A[i] = A[i - 1] + A[i - 2] * A[i - 3];
}

void kernel3(float *h, float *w, int *idx, int size)
{
    for (int i = 0; i < ARRAY_SIZE; ++i)
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
    for (i = 0; i < size - size % 5; i++)
    {
        // removing the if statment
        diff = A[i] - B[i];
        sum = (sum + diff * (diff > 0));
        diff = A[i + 1] - B[i + 1];
        sum = (sum + diff * (diff > 0));
        diff = A[i + 2] - B[i + 2];
        sum = (sum + diff * (diff > 0));
        diff = A[i + 3] - B[i + 3];
        sum = (sum + diff * (diff > 0));
        diff = A[i + 4] - B[i + 4];
    }
    i++;
    if (i < size)
        diff = A[i] - B[i];
    sum = (sum + diff * (diff > 0));
    i++;
    if (i < size)
        diff = A[i] - B[i];
    sum = (sum + diff * (diff > 0));
    i++;
    if (i < size)
        diff = A[i] - B[i];
    sum = (sum + diff * (diff > 0));
    i++;
    if (i < size)
        diff = A[i] - B[i];
    sum = (sum + diff * (diff > 0));
    i++;
    if (i < size)
        diff = A[i] - B[i];
    sum = (sum + diff * (diff > 0));
    return sum;
}

int main()
{

    srand(42);

    if (COLS_A != ROWS_B)
    {
        xil_printf("COLS_A=%d and ROWS_B=%d do not match.\r\n", COLS_A, ROWS_B);
        return -1;
    }

    int *A = (int *)malloc(ROWS_A * COLS_A * sizeof(int));
    int *B = (int *)malloc(ROWS_B * COLS_B * sizeof(int));
    int *C = (int *)malloc(ROWS_A * COLS_B * sizeof(int));

    if (A == NULL || B == NULL || C == NULL)
    {
        xil_printf("Memory allocation failed\n");
        return 1;
    }

    fill_matrix_random(A, ROWS_A, COLS_A);
    fill_matrix_random(B, ROWS_B, COLS_B);
    fill_matrix_zeros(C, ROWS_A, COLS_B);

    // Perform matrix multiplication
    start_timer();
    naive_matrix_multiply(A, B, C, ROWS_A, COLS_A, COLS_B);
    double t = stop_timer();

    char s[128] = {};
    sprintf(s, "Time: %.4fs\r\n", t);
    xil_printf("%s", s);

    while (1)
        naive_matrix_multiply(A, B, C, ROWS_A, COLS_A, COLS_B);

    // Free allocated memory
    free(A);
    free(B);
    free(C);
    return 0;
}
