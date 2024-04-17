#include<stdlib.h>
#include<time.h>
#include<stdio.h>
#include<cuda_profiler_api.h>
#include <cuda_runtime.h>

#define IMAGE_W 5000
#define IMAGE_H 5000
#define BLOCK_SIZE 32

// Forward declaration of the kernel function
__global__ void SobelFilter(int*, int*);

// Kernel (Runs on Device)
__global__ void SobelFilter(int* dataIn, int* dataOut)
{

    // Define Sobel masks
    const int sobel_mask_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}};
    const int sobel_mask_y[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}};

    // Thread index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < IMAGE_H - 1 && column < IMAGE_W - 1 && row > 0 && column > 0) {
        int gx = 0, gy = 0;
        // Apply Sobel masks to compute gradients
        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {
                gx += sobel_mask_x[i + 1][j + 1] * dataIn[(row + i) * IMAGE_W + (column + j)];
                gy += sobel_mask_y[i + 1][j + 1] * dataIn[(row + i) * IMAGE_W + (column + j)];
            }
        }
        // Compute gradient magnitude
       //printf("gx: %d \n", gx);
        double absSqrt = sqrt((double)(gx * gx + gy * gy));
        // Assign gradient magnitude to output pixel
        dataOut[row * IMAGE_W + column] = (int)absSqrt;
    }
}

int main()
{
    // Step 1: Allocate Host Memory
   int dataIn[IMAGE_H * IMAGE_W];
   int dataOut[IMAGE_H * IMAGE_W];

    // Read pixel values from the input file
    FILE* FD = fopen("input.txt", "r");
    if (FD == NULL)
    {
        perror("Error opening input file");
        return 1;
    }
    for (int i = 0; i < IMAGE_H; i++)
    {
        for (int j = 0; j < IMAGE_W; j++)
        {
            unsigned char pixel;
            if (fscanf(FD, "%hhu", &pixel) != 1)
            {
                perror("Error reading from input file");
                fclose(FD);
                return 1;
            }
            // Store pixel value in input data array
            dataIn[i * IMAGE_W + j] = (int)pixel;
            //printf("data read is %d\n",(int)dataIn[i * IMAGE_W + j]);
        }
    }

    fclose(FD);

    // Clock for initialization
    clock_t before_init = clock();
    cudaProfilerStart();

    // Step 2: Allocate device memory for A and B
    int *d_dataIn, *d_dataOut;
    cudaMalloc(&d_dataIn, IMAGE_H * IMAGE_W * sizeof(int));
    cudaMalloc(&d_dataOut, IMAGE_H * IMAGE_W * sizeof(int));

    // Step 3: Copy data from host memory to device memory
    cudaEvent_t start_memcpyh2d, stop_memcpyh2d;
    cudaEventCreate(&start_memcpyh2d);
    cudaEventCreate(&stop_memcpyh2d);
    cudaEventRecord(start_memcpyh2d);
    cudaMemcpy(d_dataIn, dataIn, IMAGE_H * IMAGE_W * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(stop_memcpyh2d);
    cudaEventSynchronize(stop_memcpyh2d);

    float mseconds1 = 0.0;
    cudaEventElapsedTime(&mseconds1, start_memcpyh2d, stop_memcpyh2d);
    printf("Time of the MEMCPY of %d bytes: %2.3f ms\n", (IMAGE_H * IMAGE_W * sizeof(int), mseconds1));

// Setup the execution configuration (grid size and block size)
dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); // 32x32 threads per block
dim3 dimGrid((IMAGE_W + dimBlock.x - 1) / dimBlock.x, (IMAGE_H + dimBlock.y - 1) / dimBlock.y);

// Step 4: Launch the device computation
cudaEvent_t start_kernel, stop_kernel;
cudaEventCreate(&start_kernel);
cudaEventCreate(&stop_kernel);
cudaEventRecord(start_kernel);
SobelFilter<<<dimGrid, dimBlock>>>(d_dataIn, d_dataOut);
cudaEventRecord(stop_kernel);
cudaEventSynchronize(stop_kernel);

    float mksec = 0.0;
    cudaEventElapsedTime(&mksec, start_kernel, stop_kernel);
    printf("Time to complete CUDA Sobel kernel %d size: %2.3f ms\n", IMAGE_W, mksec);

    // Step 5: Read results from the device
    cudaEvent_t start_memcpyd2h, stop_memcpyd2h;
    cudaEventCreate(&start_memcpyd2h);
    cudaEventCreate(&stop_memcpyd2h);
    cudaEventRecord(start_memcpyd2h);
    cudaMemcpy(dataOut, d_dataOut, IMAGE_H * IMAGE_W * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_memcpyd2h);
    cudaEventSynchronize(stop_memcpyd2h);

    float mseconds2 = 0.0;
    cudaEventElapsedTime(&mseconds2, start_memcpyd2h, stop_memcpyd2h);
    printf("Time of the MEMCPY of %d bytes: %2.3f ms\n", (IMAGE_H * IMAGE_W * sizeof(int)), mseconds2);

    // Clock for initialization
    clock_t after_init = clock();
    printf("Execution time for initialization (msec) = %ld\n", ((after_init - before_init) * 1000) / CLOCKS_PER_SEC);

    // Step 6: Write processed image to output file
    FD = fopen("output.txt", "w");
    if (FD == NULL)
    {
        perror("Error opening output file");
        return 1;
    }
    for (int i = 0; i < IMAGE_H; i++)
    {
        for (int j = 0; j < IMAGE_W; j++)
        {
            //printf("data out is %d\n",dataOut[i * IMAGE_W + j]);
            unsigned char pixel = (unsigned char)dataOut[i * IMAGE_W + j];
            // Write pixel value to file
            fprintf(FD, "%hhu\t",pixel);

        }
        // Write newline after each row
        fprintf(FD, "\n");
    }
    fclose(FD);

    // Step 7: Free device memory
    cudaFree(d_dataIn);
    cudaFree(d_dataOut);

    return 0;
}
