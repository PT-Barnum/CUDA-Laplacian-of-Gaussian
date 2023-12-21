#include <stdio.h>
#include "image.h"
extern "C"

__device__ void getPixelDevice(unsigned char* image, int width, int height, int channels, int x, int y, Pixel *ret_pixel)
{
    unsigned char* pixel = &image[channels * (y*width + x)];
    ret_pixel->red = pixel[0];
    ret_pixel->green = pixel[1];
    ret_pixel->blue = pixel[2];
}

__device__ void setPixelDevice(unsigned char *output, Pixel *output_pixel, int width, int height, int channels, int x, int y)
{
    unsigned char* set_pixel = &output[channels * (y*width + x)];
    set_pixel[0] = output_pixel->red;
    set_pixel[1] = output_pixel->green;
    set_pixel[2] = output_pixel->blue;
}

__device__ void resetPixelDevice(Pixel *pixel)
{
    pixel->red = 0;
    pixel->green = 0;
    pixel->blue = 0;
}

__device__ float relativeLuminanceDevice(Pixel *pixel)
{
    // L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    float red_value = (float)pixel->red * 0.2126;
    float green_value = (float)pixel->green * 0.7152;
    float blue_value = (float)pixel->blue * 0.0722;
    return ((red_value) + (green_value) + (blue_value));
}

__global__ void grayscaleKernel(unsigned char* input, unsigned char* output, int width, int height, int channels)
{
    Pixel output_pixel;
    double luminance;

    int Row = threadIdx.x + blockIdx.x * blockDim.x;
    int Col = threadIdx.y + blockIdx.y * blockDim.y;

    if ((Row >= 0) && (Row < width) && (Col >= 0) && (Col < height))
    {
        getPixelDevice(input, width, height, channels, Row, Col, &output_pixel);
        luminance = relativeLuminanceDevice(&output_pixel);
        output_pixel.red = (unsigned char) ((int)luminance);
        output_pixel.green = (unsigned char) ((int)luminance);
        output_pixel.blue = (unsigned char) ((int)luminance);
        setPixelDevice(output, &output_pixel, width, height, channels, Row, Col);
    }
}

stbi_uc* grayscaleCuda(stbi_uc* input, int width, int height, int channels)
{
    int image_size = width*height*channels;
    stbi_uc* output = (stbi_uc*) malloc(sizeof(stbi_uc)*image_size);

    unsigned char* input_d;
    unsigned char* output_d;
    cudaMalloc((void**)&input_d, sizeof(unsigned char)*image_size);
    cudaMalloc((void**)&output_d, sizeof(unsigned char)*image_size);
    cudaMemcpy(input_d, input, sizeof(unsigned char)*image_size, cudaMemcpyHostToDevice);

    int x_blocks = (width % GRAYSCALE_BLOCK_SIZE) ? (width / GRAYSCALE_BLOCK_SIZE) + 1 : (width / GRAYSCALE_BLOCK_SIZE);
    int y_blocks = (height % GRAYSCALE_BLOCK_SIZE) ? (height / GRAYSCALE_BLOCK_SIZE) + 1 : (height / GRAYSCALE_BLOCK_SIZE);
    dim3 DimGrid(x_blocks, y_blocks);
    dim3 DimBlock(GRAYSCALE_BLOCK_SIZE, GRAYSCALE_BLOCK_SIZE);

    grayscaleKernel<<<DimGrid, DimBlock>>>(input_d, output_d, width, height, channels);

    cudaMemcpy(output, output_d, sizeof(unsigned char)*image_size, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
    return output;
}

__device__ int laplacianCoordinate(int inner, int outer)
{
    if (inner == outer-1)
    {
        return 0;
    }
    else if (inner == outer)
    {
        return 1;
    }
    else
    {
        return 2;
    }
}

__global__ void laplacianKernelNoOptimizations(unsigned char* input, unsigned char* output, 
                            int width, int height, int channels, int* laplacian_convolution)
{
    int Row = threadIdx.x + blockIdx.x * blockDim.x;
    int Col = threadIdx.y + blockIdx.y * blockDim.y;

    if ((Row >= 0) && (Row < width) && (Col >= 0) && (Col < height))
    {
        Pixel current_pixel;
        Pixel output_pixel;
        float luminance_value = 0.0f;
        resetPixelDevice(&current_pixel);
        resetPixelDevice(&output_pixel);
        int x_;
        int y_;
        for (int x=Row-1; x<Row+2; x++)
        {
            for (int y=Col-1; y<Col+2; y++)
            {
                if ((x >= 0) && (x < width) && (y >= 0) && (y < height))
                {
                    getPixelDevice(input, width, height, channels, x, y, &current_pixel);
                    x_ = laplacianCoordinate(x, Row);
                    y_ = laplacianCoordinate(y, Col);
                    luminance_value = luminance_value + 
                            ((relativeLuminanceDevice(&current_pixel)) * laplacian_convolution[y_*LAPLACIAN_KERNEL_SIZE+x_]);
                }
            }
        }
        output_pixel.red = luminance_value;
        output_pixel.green = luminance_value;
        output_pixel.blue = luminance_value;
        setPixelDevice(output, &output_pixel, width, height, channels, Row, Col);
    }
}

stbi_uc* laplacianCudaNoOptimizations(stbi_uc* input, int width, int height, int channels)
{
    int image_size = width*height*channels;
    stbi_uc* output = (stbi_uc*) malloc(sizeof(stbi_uc)*image_size);
    unsigned char* input_d;
    unsigned char* output_d;
    cudaMalloc((void**)&input_d, sizeof(unsigned char)*image_size);
    cudaMalloc((void**)&output_d, sizeof(unsigned char)*image_size);
    cudaMemcpy(input_d, input, sizeof(unsigned char)*image_size, cudaMemcpyHostToDevice);

    int convolution[LAPLACIAN_KERNEL_SIZE*LAPLACIAN_KERNEL_SIZE] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
    int* convolution_d;
    cudaMalloc((void**)&convolution_d, sizeof(int)*LAPLACIAN_KERNEL_SIZE*LAPLACIAN_KERNEL_SIZE);
    cudaMemcpy(convolution_d, convolution, sizeof(int)*LAPLACIAN_KERNEL_SIZE*LAPLACIAN_KERNEL_SIZE, cudaMemcpyHostToDevice);

    int x_blocks = (width % LAPLACIAN_BLOCK_SIZE) ? ((width / LAPLACIAN_BLOCK_SIZE) + 1) : (width / LAPLACIAN_BLOCK_SIZE);
    int y_blocks = (height % LAPLACIAN_BLOCK_SIZE) ? ((height / LAPLACIAN_BLOCK_SIZE) + 1) : (height / LAPLACIAN_BLOCK_SIZE);

    dim3 DimGrid(x_blocks, y_blocks);
    dim3 DimBlock(LAPLACIAN_BLOCK_SIZE, LAPLACIAN_BLOCK_SIZE);

    laplacianKernelNoOptimizations<<<DimGrid, DimBlock>>>(input_d, output_d, width, height, channels, convolution_d);

    cudaMemcpy(output, output_d, sizeof(unsigned char)*image_size, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
    cudaFree(convolution_d);
    return output;
}

__constant__ int laplacian_convolution[LAPLACIAN_KERNEL_SIZE*LAPLACIAN_KERNEL_SIZE];

__global__ void laplacianKernelConstant(unsigned char* input, unsigned char* output, int width, int height, int channels)
{
    int Row = threadIdx.x + blockIdx.x * blockDim.x;
    int Col = threadIdx.y + blockIdx.y * blockDim.y;

    if ((Row >= 0) && (Row < width) && (Col >= 0) && (Col < height))
    {
        Pixel current_pixel;
        Pixel output_pixel;
        float luminance_value = 0.0f;
        resetPixelDevice(&current_pixel);
        resetPixelDevice(&output_pixel);
        int x_;
        int y_;
        for (int x=Row-1; x<Row+2; x++)
        {
            for (int y=Col-1; y<Col+2; y++)
            {
                if ((x >= 0) && (x < width) && (y >= 0) && (y < height))
                {
                    getPixelDevice(input, width, height, channels, x, y, &current_pixel);
                    x_ = laplacianCoordinate(x, Row);
                    y_ = laplacianCoordinate(y, Col);
                    luminance_value = luminance_value + 
                            ((relativeLuminanceDevice(&current_pixel)) * laplacian_convolution[y_*LAPLACIAN_KERNEL_SIZE+x_]);
                }
            }
        }
        output_pixel.red = luminance_value;
        output_pixel.green = luminance_value;
        output_pixel.blue = luminance_value;
        setPixelDevice(output, &output_pixel, width, height, channels, Row, Col);
    }
}

stbi_uc* laplacianCudaConstant(stbi_uc* input, int width, int height, int channels)
{
    int image_size = width*height*channels;
    stbi_uc* output = (stbi_uc*) malloc(sizeof(stbi_uc)*image_size);
    unsigned char* input_d;
    unsigned char* output_d;
    cudaMalloc((void**)&input_d, sizeof(unsigned char)*image_size);
    cudaMalloc((void**)&output_d, sizeof(unsigned char)*image_size);
    cudaMemcpy(input_d, input, sizeof(unsigned char)*image_size, cudaMemcpyHostToDevice);

    int convolution[LAPLACIAN_KERNEL_SIZE*LAPLACIAN_KERNEL_SIZE] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
    cudaMemcpyToSymbol(laplacian_convolution, convolution, sizeof(int)*LAPLACIAN_KERNEL_SIZE*LAPLACIAN_KERNEL_SIZE);

    int x_blocks = (width % LAPLACIAN_BLOCK_SIZE) ? ((width / LAPLACIAN_BLOCK_SIZE) + 1) : (width / LAPLACIAN_BLOCK_SIZE);
    int y_blocks = (height % LAPLACIAN_BLOCK_SIZE) ? ((height / LAPLACIAN_BLOCK_SIZE) + 1) : (height / LAPLACIAN_BLOCK_SIZE);

    dim3 DimGrid(x_blocks, y_blocks);
    dim3 DimBlock(LAPLACIAN_BLOCK_SIZE, LAPLACIAN_BLOCK_SIZE);

    laplacianKernelConstant<<<DimGrid, DimBlock>>>(input_d, output_d, width, height, channels);

    cudaMemcpy(output, output_d, sizeof(unsigned char)*image_size, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
    return output;
}


__global__ void laplacianKernelConstantShared(unsigned char* input, unsigned char* output, int width, int height, int channels)
{
    __shared__ float tile_luminances[LAPLACIAN_BLOCK_SIZE][LAPLACIAN_BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = ty + blockIdx.y * LAPLACIAN_TILE_SIZE;
    int col_o = tx + blockIdx.x * LAPLACIAN_TILE_SIZE;
    Pixel current_pixel;
    Pixel output_pixel;

    int n = (LAPLACIAN_KERNEL_SIZE >> 1);
    int row_i = row_o - n;
    int col_i = col_o - n;

    if(row_i >= 0 && row_i < height && col_i >= 0 && col_i < width)
    {
        getPixelDevice(input, width, height, channels, col_i, row_i, &current_pixel);
        tile_luminances[ty][tx] = (relativeLuminanceDevice(&current_pixel));
    }   
    else
    {
        tile_luminances[ty][tx] = 0;
    }
    __syncthreads();

    if(tx < LAPLACIAN_TILE_SIZE && ty < LAPLACIAN_TILE_SIZE)
    {
        float luminance_value = 0.0f;
        for(int y=0; y<LAPLACIAN_KERNEL_SIZE; y++)
        {
            for(int x=0; x<LAPLACIAN_KERNEL_SIZE; x++)
            {
                luminance_value = luminance_value + (laplacian_convolution[y*LAPLACIAN_KERNEL_SIZE + x] * tile_luminances[y+ty][x+tx]);
            }
        }
        
        if(row_o < height && col_o < width)
        {
            output_pixel.red = (unsigned char) (luminance_value);
            output_pixel.green = (unsigned char) (luminance_value);
            output_pixel.blue = (unsigned char) (luminance_value);
            setPixelDevice(output, &output_pixel, width, height, channels, col_o, row_o);
        }
    }

    __syncthreads();
}

stbi_uc* laplacianCudaConstantShared(stbi_uc* input, int width, int height, int channels)
{
    int image_size = width*height*channels;
    stbi_uc* output = (stbi_uc*) malloc(sizeof(stbi_uc)*image_size);
    unsigned char* input_d;
    unsigned char* output_d;
    cudaMalloc((void**)&input_d, sizeof(unsigned char)*image_size);
    cudaMalloc((void**)&output_d, sizeof(unsigned char)*image_size);
    cudaMemcpy(input_d, input, sizeof(unsigned char)*image_size, cudaMemcpyHostToDevice);

    int convolution[LAPLACIAN_KERNEL_SIZE*LAPLACIAN_KERNEL_SIZE] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
    cudaMemcpyToSymbol(laplacian_convolution, convolution, LAPLACIAN_KERNEL_SIZE*LAPLACIAN_KERNEL_SIZE*sizeof(int));

    int x_blocks = (width % LAPLACIAN_TILE_SIZE) ? ((width / LAPLACIAN_TILE_SIZE) + 1) : (width / LAPLACIAN_TILE_SIZE);
    int y_blocks = (height % LAPLACIAN_TILE_SIZE) ? ((height / LAPLACIAN_TILE_SIZE) + 1) : (height / LAPLACIAN_TILE_SIZE);

    dim3 DimGrid(x_blocks, y_blocks);
    dim3 DimBlock(LAPLACIAN_BLOCK_SIZE, LAPLACIAN_BLOCK_SIZE);

    laplacianKernelConstantShared<<<DimGrid, DimBlock>>>(input_d, output_d, width, height, channels);

    cudaMemcpy(output, output_d, sizeof(unsigned char)*image_size, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
    return output;
}

__constant__ int gaussian_convolution[GAUSSIAN_KERNEL_SIZE*GAUSSIAN_KERNEL_SIZE];

__device__ int gaussianCoordinate(int inner, int outer)
{
    if (inner == outer-2)
        return 0;
    else if (inner == outer-1)
        return 1;
    else if (inner == outer)
        return 2;
    else if (inner == outer+1)
        return 3;
    else
        return 4;
}

__global__ void gaussianKernelNoOptimizations(stbi_uc* input, stbi_uc* output, int width, int height, int channels, int* global_convolution)
{
    int Row = threadIdx.x + blockIdx.x * blockDim.x;
    int Col = threadIdx.y + blockIdx.y * blockDim.y;

    if ((Row >= 0) && (Row < width) && (Col >= 0) && (Col < height))
    {
        Pixel current_pixel;
        Pixel output_pixel;
        float luminance_value = 0.0f;
        resetPixelDevice(&current_pixel);
        resetPixelDevice(&output_pixel);
        int x_;
        int y_;
        for (int x=Row-2; x<Row+3; x++)
        {
            for (int y=Col-2; y<Col+3; y++)
            {
                if ((x >= 0) && (x < width) && (y >= 0) && (y < height))
                {
                    getPixelDevice(input, width, height, channels, x, y, &current_pixel);
                    x_ = gaussianCoordinate(x, Row);
                    y_ = gaussianCoordinate(y, Col);
                    luminance_value = luminance_value + 
                            ((relativeLuminanceDevice(&current_pixel)) * global_convolution[y_*GAUSSIAN_KERNEL_SIZE+x_]);
                }
            }
        }
        output_pixel.red = (unsigned char) (luminance_value / 273);
        output_pixel.green = (unsigned char) (luminance_value / 273);
        output_pixel.blue = (unsigned char) (luminance_value / 273);
        setPixelDevice(output, &output_pixel, width, height, channels, Row, Col);
    }
}

stbi_uc* gaussianCudaNoOptimizations(stbi_uc* input, int width, int height, int channels)
{
    int image_size = width * height * channels;
    stbi_uc* output = (stbi_uc*) malloc(sizeof(stbi_uc)*image_size);
    unsigned char* input_d;
    unsigned char* output_d;
    int* convolution_d;

    cudaMalloc((void**)&input_d, sizeof(unsigned char)*image_size);
    cudaMalloc((void**)&output_d, sizeof(unsigned char)*image_size);
    cudaMalloc((void**)&convolution_d, sizeof(int)*GAUSSIAN_KERNEL_SIZE*GAUSSIAN_KERNEL_SIZE);
    cudaMemcpy(input_d, input, sizeof(unsigned char)*image_size, cudaMemcpyHostToDevice);

    // int convolution[] = {1, 4, 7, 4, 1, 4, 16, 26, 16, 4, 7, 26, 41, 26, 7, 4, 16, 26, 16, 4, 1, 4, 7, 4, 1};
    int convolution[] = {1,  4,  6,  4,  1,
                         4, 16, 24, 16,  4,
                         6, 24, 36, 24,  6,
                         4, 16, 24, 16,  4,
                         1,  4,  6,  4,  1};
    cudaMemcpy(convolution_d, convolution, sizeof(int)*GAUSSIAN_KERNEL_SIZE*GAUSSIAN_KERNEL_SIZE, cudaMemcpyHostToDevice);

    int x_blocks = (width % GAUSSIAN_BLOCK_SIZE) ? ((width / GAUSSIAN_BLOCK_SIZE) + 1) : (width / GAUSSIAN_BLOCK_SIZE);
    int y_blocks = (height % GAUSSIAN_BLOCK_SIZE) ? ((height / GAUSSIAN_BLOCK_SIZE) + 1) : (height / GAUSSIAN_BLOCK_SIZE);

    dim3 DimGrid(x_blocks, y_blocks);
    dim3 DimBlock(GAUSSIAN_BLOCK_SIZE, GAUSSIAN_BLOCK_SIZE);

    gaussianKernelNoOptimizations<<<DimGrid, DimBlock>>>(input_d, output_d, width, height, channels, convolution_d);
    cudaDeviceSynchronize();

    cudaMemcpy(output, output_d, sizeof(unsigned char)*image_size, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
    cudaFree(convolution_d);

    return output;
}


__global__ void gaussianKernelConstant(stbi_uc* input, stbi_uc* output, int width, int height, int channels)
{
    int Row = threadIdx.x + blockIdx.x * blockDim.x;
    int Col = threadIdx.y + blockIdx.y * blockDim.y;

    if ((Row >= 0) && (Row < width) && (Col >= 0) && (Col < height))
    {
        Pixel current_pixel;
        Pixel output_pixel;
        float luminance_value = 0.0f;
        resetPixelDevice(&current_pixel);
        resetPixelDevice(&output_pixel);
        int x_;
        int y_;
        for (int x=Row-2; x<Row+3; x++)
        {
            for (int y=Col-2; y<Col+3; y++)
            {
                if ((x >= 0) && (x < width) && (y >= 0) && (y < height))
                {
                    getPixelDevice(input, width, height, channels, x, y, &current_pixel);
                    x_ = gaussianCoordinate(x, Row);
                    y_ = gaussianCoordinate(y, Col);
                    luminance_value = luminance_value + 
                            ((relativeLuminanceDevice(&current_pixel)) * gaussian_convolution[y_*GAUSSIAN_KERNEL_SIZE+x_]);
                }
            }
        }
        output_pixel.red = (unsigned char) (luminance_value / 273);
        output_pixel.green = (unsigned char) (luminance_value / 273);
        output_pixel.blue = (unsigned char) (luminance_value / 273);
        setPixelDevice(output, &output_pixel, width, height, channels, Row, Col);
    }
}

stbi_uc* gaussianCudaConstant(stbi_uc* input, int width, int height, int channels)
{
    int image_size = width * height * channels;
    stbi_uc* output = (stbi_uc*) malloc(sizeof(stbi_uc)*image_size);
    unsigned char* input_d;
    unsigned char* output_d;

    cudaMalloc((void**)&input_d, sizeof(unsigned char)*image_size);
    cudaMalloc((void**)&output_d, sizeof(unsigned char)*image_size);
    cudaMemcpy(input_d, input, sizeof(unsigned char)*image_size, cudaMemcpyHostToDevice);

    // int convolution[] = {1, 4, 7, 4, 1, 4, 16, 26, 16, 4, 7, 26, 41, 26, 7, 4, 16, 26, 16, 4, 1, 4, 7, 4, 1};
    int convolution[] = {1,  4,  6,  4,  1,
                         4, 16, 24, 16,  4,
                         6, 24, 36, 24,  6,
                         4, 16, 24, 16,  4,
                         1,  4,  6,  4,  1};
    cudaMemcpyToSymbol(gaussian_convolution, convolution, sizeof(int)*GAUSSIAN_KERNEL_SIZE*GAUSSIAN_KERNEL_SIZE);

    int x_blocks = (width % GAUSSIAN_BLOCK_SIZE) ? ((width / GAUSSIAN_BLOCK_SIZE) + 1) : (width / GAUSSIAN_BLOCK_SIZE);
    int y_blocks = (height % GAUSSIAN_BLOCK_SIZE) ? ((height / GAUSSIAN_BLOCK_SIZE) + 1) : (height / GAUSSIAN_BLOCK_SIZE);

    dim3 DimGrid(x_blocks, y_blocks);
    dim3 DimBlock(GAUSSIAN_BLOCK_SIZE, GAUSSIAN_BLOCK_SIZE);

    gaussianKernelConstant<<<DimGrid, DimBlock>>>(input_d, output_d, width, height, channels);
    cudaDeviceSynchronize();

    cudaMemcpy(output, output_d, sizeof(unsigned char)*image_size, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);

    return output;
}


__global__ void gaussianKernelConstantShared(unsigned char* input, unsigned char* output, int width, int height, int channels)
{
    __shared__ double tile_luminances[GAUSSIAN_BLOCK_SIZE][GAUSSIAN_BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = ty + blockIdx.y * GAUSSIAN_TILE_SIZE;
    int col_o = tx + blockIdx.x * GAUSSIAN_TILE_SIZE;
    Pixel current_pixel;
    Pixel output_pixel;

    int n = (GAUSSIAN_KERNEL_SIZE >> 1);
    int row_i = row_o - n;
    int col_i = col_o - n;

    resetPixelDevice(&current_pixel);
    resetPixelDevice(&output_pixel);
    if(row_i >= 0 && row_i < height && col_i >= 0 && col_i < width)
    {
        getPixelDevice(input, width, height, channels, col_i, row_i, &current_pixel);
        tile_luminances[ty][tx] = relativeLuminanceDevice(&current_pixel);
    }   
    else
        tile_luminances[ty][tx] = 0;

    __syncthreads();

    if(tx < GAUSSIAN_TILE_SIZE && ty < GAUSSIAN_TILE_SIZE){
        float luminance_value = 0.0f;
        for(int y=0; y<GAUSSIAN_KERNEL_SIZE; y++)
            for(int x=0; x<GAUSSIAN_KERNEL_SIZE; x++)
            {
                luminance_value += gaussian_convolution[y*GAUSSIAN_KERNEL_SIZE + x] * tile_luminances[y+ty][x+tx];
            }

        if(row_o < height && col_o < width)
        {
            output_pixel.red = (unsigned char) (luminance_value / 273);
            output_pixel.green = (unsigned char) (luminance_value / 273);
            output_pixel.blue = (unsigned char) (luminance_value / 273);
            setPixelDevice(output, &output_pixel, width, height, channels, col_o, row_o);
        }
    }

    __syncthreads();
}

stbi_uc* gaussianCudaConstantShared(stbi_uc* input, int width, int height, int channels)
{
    int image_size = width * height * channels;
    stbi_uc* output = (stbi_uc*) malloc(sizeof(stbi_uc)*image_size);
    unsigned char* input_d;
    unsigned char* output_d;

    cudaMalloc((void**)&input_d, sizeof(unsigned char)*image_size);
    cudaMalloc((void**)&output_d, sizeof(unsigned char)*image_size);
    cudaMemcpy(input_d, input, sizeof(unsigned char)*image_size, cudaMemcpyHostToDevice);

    // int convolution[] = {1, 4, 7, 4, 1, 4, 16, 26, 16, 4, 7, 26, 41, 26, 7, 4, 16, 26, 16, 4, 1, 4, 7, 4, 1};
    int convolution[] = {1,  4,  6,  4,  1,
                         4, 16, 24, 16,  4,
                         6, 24, 36, 24,  6,
                         4, 16, 24, 16,  4,
                         1,  4,  6,  4,  1};
    cudaMemcpyToSymbol(gaussian_convolution, convolution, 5*5*sizeof(int));

    // Trying with __constant__ and __shared__ memories technique
    int x_blocks = (width % GAUSSIAN_TILE_SIZE) ? ((width / GAUSSIAN_TILE_SIZE) + 1) : (width / GAUSSIAN_TILE_SIZE);
    int y_blocks = (height % GAUSSIAN_TILE_SIZE) ? ((height / GAUSSIAN_TILE_SIZE) + 1) : (height / GAUSSIAN_TILE_SIZE);

    dim3 DimGrid(x_blocks, y_blocks);
    dim3 DimBlock(GAUSSIAN_BLOCK_SIZE, GAUSSIAN_BLOCK_SIZE);

    gaussianKernelConstantShared<<<DimGrid, DimBlock>>>(input_d, output_d, width, height, channels);
    cudaDeviceSynchronize();

    cudaMemcpy(output, output_d, sizeof(unsigned char)*image_size, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);

    return output;
}

__device__ int laplacianGaussianCoordinate(int inner, int outer)
{
if (inner == outer-4)
    {
        return 0;
    }
    else if (inner == outer-3)
    {
        return 1;
    }
    else if (inner == outer-2)
    {
        return 2;
    }
    else if (inner == outer-1)
    {
        return 3;
    }
    else if (inner == outer)
    {
        return 4;
    }
    else if (inner == outer+1)
    {
        return 5;
    }
    else if (inner == outer+2)
    {
        return 6;
    }
    else if (inner == outer+3)
    {
        return 7;
    }
    else
    {
        return 8;
    }
}

__global__ void laplacianGaussianKernelNoOptimizations(unsigned char* input, unsigned char* output, int width, int height, int channels, int* convolution)
{
    int Row = threadIdx.x + blockIdx.x * blockDim.x;
    int Col = threadIdx.y + blockIdx.y * blockDim.y;

    if ((Row >= 0) && (Row < width) && (Col >= 0) && (Col < height))
    {
        Pixel current_pixel;
        Pixel output_pixel;
        float luminance_value = 0.0f;
        resetPixelDevice(&current_pixel);
        resetPixelDevice(&output_pixel);
        int x_;
        int y_;
        for (int x=Row-4; x<Row+5; x++)
        {
            for (int y=Col-4; y<Col+5; y++)
            {
                if ((x >= 0) && (x < width) && (y >= 0) && (y < height))
                {
                    getPixelDevice(input, width, height, channels, x, y, &current_pixel);
                    x_ = laplacianGaussianCoordinate(x, Row);
                    y_ = laplacianGaussianCoordinate(y, Col);
                    luminance_value = luminance_value + 
                            ((relativeLuminanceDevice(&current_pixel)) * convolution[y_*LOG_KERNEL_SIZE+x_]);
                }
            }
        }
        output_pixel.red = luminance_value;
        output_pixel.green = luminance_value;
        output_pixel.blue = luminance_value;
        setPixelDevice(output, &output_pixel, width, height, channels, Row, Col);
    }
}

stbi_uc* laplacianGaussianCudaNoOptimizations(stbi_uc* input, int width, int height, int channels)
{
    int image_size = width * height * channels;
    stbi_uc* output = (stbi_uc*) malloc(sizeof(stbi_uc)*image_size);
    unsigned char* input_d;
    unsigned char* output_d;
    int* convolution_d;

    cudaMalloc((void**)&input_d, sizeof(unsigned char)*image_size);
    cudaMalloc((void**)&output_d, sizeof(unsigned char)*image_size);
    cudaMalloc((void**)&convolution_d, sizeof(int)*LOG_KERNEL_SIZE*LOG_KERNEL_SIZE);
    cudaMemcpy(input_d, input, sizeof(unsigned char)*image_size, cudaMemcpyHostToDevice);

    int convolution[] = {0, 1, 1, 2, 2, 2, 1, 1, 0,
                        1, 2, 4, 5, 5, 5, 4, 2, 1,
                        1, 4, 5, 3, 0, 3, 5, 4, 1,
                        2, 5, 3, -12, -24, -12, 3, 5, 2,
                        2, 5, 0, -24, -40, -24, 0, 5, 2,
                        2, 5, 3, -12, -24, -12, 3, 5, 2,
                        1, 4, 5, 3, 0, 3, 5, 4, 1,
                        1, 2, 4, 5, 5, 5, 4, 2, 1,
                        0, 1, 1, 2, 2, 2, 1, 1, 0};
    cudaMemcpy(convolution_d, convolution, sizeof(int)*LOG_KERNEL_SIZE*LOG_KERNEL_SIZE, cudaMemcpyHostToDevice);

    int x_blocks = (width % LOG_BLOCK_SIZE) ? ((width / LOG_BLOCK_SIZE) + 1) : (width / LOG_BLOCK_SIZE);
    int y_blocks = (height % LOG_BLOCK_SIZE) ? ((height / LOG_BLOCK_SIZE) + 1) : (height / LOG_BLOCK_SIZE);

    dim3 DimGrid(x_blocks, y_blocks);
    dim3 DimBlock(LOG_BLOCK_SIZE, LOG_BLOCK_SIZE);

    laplacianGaussianKernelNoOptimizations<<<DimGrid, DimBlock>>>(input_d, output_d, width, height, channels, convolution_d);
    cudaDeviceSynchronize();

    cudaMemcpy(output, output_d, sizeof(unsigned char)*image_size, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
    cudaFree(convolution_d);

    return output;
}

__constant__ int laplacian_gaussian_convolution[LOG_KERNEL_SIZE*LOG_KERNEL_SIZE];

__global__ void laplacianGaussianKernelConstant(unsigned char* input, unsigned char* output, int width, int height, int channels)
{
    int Row = threadIdx.x + blockIdx.x * blockDim.x;
    int Col = threadIdx.y + blockIdx.y * blockDim.y;

    if ((Row >= 0) && (Row < width) && (Col >= 0) && (Col < height))
    {
        Pixel current_pixel;
        Pixel output_pixel;
        float luminance_value = 0.0f;
        resetPixelDevice(&current_pixel);
        resetPixelDevice(&output_pixel);
        int x_;
        int y_;
        for (int x=Row-4; x<Row+5; x++)
        {
            for (int y=Col-4; y<Col+5; y++)
            {
                if ((x >= 0) && (x < width) && (y >= 0) && (y < height))
                {
                    getPixelDevice(input, width, height, channels, x, y, &current_pixel);
                    x_ = laplacianGaussianCoordinate(x, Row);
                    y_ = laplacianGaussianCoordinate(y, Col);
                    luminance_value = luminance_value + 
                            ((relativeLuminanceDevice(&current_pixel)) * laplacian_gaussian_convolution[y_*LOG_KERNEL_SIZE+x_]);
                }
            }
        }
        output_pixel.red = luminance_value;
        output_pixel.green = luminance_value;
        output_pixel.blue = luminance_value;
        setPixelDevice(output, &output_pixel, width, height, channels, Row, Col);
    }
}

stbi_uc* laplacianGaussianCudaConstant(stbi_uc* input, int width, int height, int channels)
{
    int image_size = width * height * channels;
    stbi_uc* output = (stbi_uc*) malloc(sizeof(stbi_uc)*image_size);
    unsigned char* input_d;
    unsigned char* output_d;

    cudaMalloc((void**)&input_d, sizeof(unsigned char)*image_size);
    cudaMalloc((void**)&output_d, sizeof(unsigned char)*image_size);
    cudaMemcpy(input_d, input, sizeof(unsigned char)*image_size, cudaMemcpyHostToDevice);

    int convolution[] = {0, 1, 1, 2, 2, 2, 1, 1, 0,
                        1, 2, 4, 5, 5, 5, 4, 2, 1,
                        1, 4, 5, 3, 0, 3, 5, 4, 1,
                        2, 5, 3, -12, -24, -12, 3, 5, 2,
                        2, 5, 0, -24, -40, -24, 0, 5, 2,
                        2, 5, 3, -12, -24, -12, 3, 5, 2,
                        1, 4, 5, 3, 0, 3, 5, 4, 1,
                        1, 2, 4, 5, 5, 5, 4, 2, 1,
                        0, 1, 1, 2, 2, 2, 1, 1, 0};
    cudaMemcpyToSymbol(laplacian_gaussian_convolution, convolution, sizeof(int)*LOG_KERNEL_SIZE*LOG_KERNEL_SIZE);

    int x_blocks = (width % LOG_BLOCK_SIZE) ? ((width / LOG_BLOCK_SIZE) + 1) : (width / LOG_BLOCK_SIZE);
    int y_blocks = (height % LOG_BLOCK_SIZE) ? ((height / LOG_BLOCK_SIZE) + 1) : (height / LOG_BLOCK_SIZE);

    dim3 DimGrid(x_blocks, y_blocks);
    dim3 DimBlock(LOG_BLOCK_SIZE, LOG_BLOCK_SIZE);

    laplacianGaussianKernelConstant<<<DimGrid, DimBlock>>>(input_d, output_d, width, height, channels);
    cudaDeviceSynchronize();

    cudaMemcpy(output, output_d, sizeof(unsigned char)*image_size, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);

    return output;
}


__global__ void laplacianGaussianKernelConstantShared(stbi_uc* input, stbi_uc* output, int width, int height, int channels)
{
    __shared__ float tile_luminances[LOG_BLOCK_SIZE][LOG_BLOCK_SIZE];


    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = ty + blockIdx.y * LOG_TILE_SIZE;
    int col_o = tx + blockIdx.x * LOG_TILE_SIZE;
    Pixel current_pixel;
    Pixel output_pixel;

    int n = (LOG_KERNEL_SIZE >> 1);
    int row_i = row_o - n;
    int col_i = col_o - n;

    if(row_i >= 0 && row_i < height && col_i >= 0 && col_i < width)
    {
        getPixelDevice(input, width, height, channels, col_i, row_i, &current_pixel);
        tile_luminances[ty][tx] = (relativeLuminanceDevice(&current_pixel));
    }   
    else
    {
        tile_luminances[ty][tx] = 0;
    }
    __syncthreads();

    if(tx < LOG_TILE_SIZE && ty < LOG_TILE_SIZE)
    {
        float luminance_value = 0.0f;
        for(int y=0; y<LOG_KERNEL_SIZE; y++)
        {
            for(int x=0; x<LOG_KERNEL_SIZE; x++)
            {
                luminance_value = luminance_value + (laplacian_gaussian_convolution[y*LOG_KERNEL_SIZE + x] * tile_luminances[y+ty][x+tx]);
            }
        }
        
        if(row_o < height && col_o < width)
        {
            output_pixel.red = (unsigned char) (luminance_value);
            output_pixel.green = (unsigned char) (luminance_value);
            output_pixel.blue = (unsigned char) (luminance_value);
            setPixelDevice(output, &output_pixel, width, height, channels, col_o, row_o);
        }
    }

    __syncthreads();
}


stbi_uc* laplacianGaussianCudaConstantShared(stbi_uc* input, int width, int height, int channels)
{
    int image_size = width * height * channels;
    stbi_uc* output = (stbi_uc*) malloc(sizeof(stbi_uc)*image_size);
    unsigned char* input_d;
    unsigned char* output_d;

    cudaMalloc((void**)&input_d, sizeof(unsigned char)*image_size);
    cudaMalloc((void**)&output_d, sizeof(unsigned char)*image_size);

    cudaMemcpy(input_d, input, sizeof(unsigned char)*image_size, cudaMemcpyHostToDevice);

    int convolution[] = {0, 1, 1, 2, 2, 2, 1, 1, 0,
                        1, 2, 4, 5, 5, 5, 4, 2, 1,
                        1, 4, 5, 3, 0, 3, 5, 4, 1,
                        2, 5, 3, -12, -24, -12, 3, 5, 2,
                        2, 5, 0, -24, -40, -24, 0, 5, 2,
                        2, 5, 3, -12, -24, -12, 3, 5, 2,
                        1, 4, 5, 3, 0, 3, 5, 4, 1,
                        1, 2, 4, 5, 5, 5, 4, 2, 1,
                        0, 1, 1, 2, 2, 2, 1, 1, 0};
    cudaMemcpyToSymbol(laplacian_gaussian_convolution, convolution, LOG_KERNEL_SIZE*LOG_KERNEL_SIZE*sizeof(int));

    // Trying with __constant__ and __shared__ memories technique
    int x_blocks = (width % LOG_TILE_SIZE) ? ((width / LOG_TILE_SIZE) + 1) : (width / LOG_TILE_SIZE);
    int y_blocks = (height % LOG_TILE_SIZE) ? ((height / LOG_TILE_SIZE) + 1) : (height / LOG_TILE_SIZE);

    dim3 DimGrid(x_blocks, y_blocks);
    dim3 DimBlock(LOG_BLOCK_SIZE, LOG_BLOCK_SIZE);

    laplacianGaussianKernelConstantShared<<<DimGrid, DimBlock>>>(input_d, output_d, width, height, channels);
    cudaDeviceSynchronize();

    cudaMemcpy(output, output_d, sizeof(unsigned char)*image_size, cudaMemcpyDeviceToHost);

    cudaFree(input_d);
    cudaFree(output_d);

    return output;
}

void validation(stbi_uc* serial, stbi_uc* parallel, int width, int height, int channels)
{
    int differences = 0;
    Pixel serial_pixel;
    Pixel parallel_pixel;
    
    for (int x=0; x<width; x++)
    {
        for (int y=0; y<height; y++)
        {
            getPixel(serial, width, height, channels, x, y, &serial_pixel);
            getPixel(parallel, width, height, channels, x, y, &parallel_pixel);
            if (fabs(relativeLuminance(&serial_pixel) - relativeLuminance(&parallel_pixel)) > .02)
            {
                differences += 1;
            }
        }
    }

    int dimensions = width * height;

    if ((float)(differences / dimensions) > 0.05)
    {
        printf("SERIAL AND PARALLEL COMPARISON FAILED: DIFFERENCES = %d, DIMENSIONS = %d\n\n", differences, dimensions);
    }
    else
    {
        printf("SERIAL AND PARALLEL COMPARISON PASSED\n\n");
    }
}

int compareGrayscales(const char* serial, const char* parallel)
{
    int serial_width, serial_height, serial_channels;
    int parallel_width, parallel_height, parallel_channels;
    stbi_uc* serial_image = loadImage(serial, &serial_width, &serial_height, &serial_channels);
    stbi_uc* parallel_image = loadImage(parallel, &parallel_width, &parallel_height, &parallel_channels);

    if ((serial_width != parallel_width) || (serial_height != parallel_height))
    {
        printf("GRAYSCALE IMAGES DO NOT MATCH DIMENSIONS\n");
        imageFree(serial_image);
        imageFree(parallel_image);
        return 1;
    }

    printf("\n*****EVALUATING GRAYSCALE IMAGES*****\n");
    validation(serial_image, parallel_image, serial_width, serial_height, serial_channels);

    imageFree(serial_image);
    imageFree(parallel_image);
    return 0;
}

int compareImages(const char* serial, const char* no_opts, const char* constant, const char* constant_shared, char* type_of_image)
{
    int serial_width, serial_height, serial_channels;
    int no_opts_width, no_opts_height, no_opts_channels;
    int constant_width, constant_height, constant_channels;
    int constant_shared_width, constant_shared_height, constant_shared_channels;
    stbi_uc* serial_image = loadImage(serial, &serial_width, &serial_height, &serial_channels);
    stbi_uc* no_opts_image = loadImage(no_opts, &no_opts_width, &no_opts_height, &no_opts_channels);
    stbi_uc* constant_image = loadImage(constant, &constant_width, &constant_height, &constant_channels);
    stbi_uc* constant_shared_image = loadImage(constant_shared, &constant_shared_width, &constant_shared_height, &constant_shared_channels);

    if (((serial_width != no_opts_width) || (serial_height != no_opts_height)) ||
        ((serial_width != constant_width) || (serial_height != constant_height)) ||
        ((serial_width != constant_shared_width) || (serial_height != constant_shared_height)))
    {
        printf("%s IMAGES DO NOT MATCH DIMENSIONS\n", type_of_image);
        imageFree(serial_image);
        imageFree(no_opts_image);
        imageFree(constant_image);
        imageFree(constant_shared_image);
        return 1;
    }

    printf("\n*****EVALUATING %s IMAGES*****\n", type_of_image);
    printf("***NO OPTIMIZATIONS***\n");
    validation(serial_image, no_opts_image, serial_width, serial_height, serial_channels);
    printf("***CONSTANT MEMORY***\n");
    validation(serial_image, constant_image, serial_width, serial_height, serial_channels);
    printf("***CONSTANT & SHARED MEMORY***\n");
    validation(serial_image, constant_shared_image, serial_width, serial_height, serial_channels);

    imageFree(serial_image);
    imageFree(no_opts_image);
    imageFree(constant_image);
    imageFree(constant_shared_image);

    return 0;
}

int performValidation()
{
    // Serial Files
    const char* serial_grayscale = "Outputs/grayscale.jpg";
    const char* serial_laplacian = "Outputs/laplacian.jpg";
    const char* serial_gaussian = "Outputs/gaussian.jpg";
    const char* serial_laplacian_of_gaussian = "Outputs/laplacian_of_gaussian.jpg";

    const char* parallel_grayscale = "CUDA_Outputs/grayscale_cuda.jpg";

    // Parallel Files **NO OPTIMIZATIONS**
    const char* no_opts_laplacian = "CUDA_Outputs/laplacian_no_optimizations.jpg";
    const char* no_opts_gaussian = "CUDA_Outputs/gaussian_no_optimizations.jpg";
    const char* no_opts_laplacian_of_gaussian = "CUDA_Outputs/laplacian_of_gaussian_no_optimizations.jpg";

    // Parallel Files **CONSTANT MEMORY**
    const char* const_laplacian = "CUDA_Outputs/laplacian_constant.jpg";
    const char* const_gaussian = "CUDA_Outputs/gaussian_constant.jpg";
    const char* const_laplacian_of_gaussian = "CUDA_Outputs/laplacian_of_gaussian_constant.jpg";

    // Parallel Files **CONSTANT & SHARED MEMORY**
    const char* const_shared_laplacian = "CUDA_Outputs/laplacian_constant_shared.jpg";
    const char* const_shared_gaussian = "CUDA_Outputs/gaussian_constant_shared.jpg";
    const char* const_shared_laplacian_of_gaussian = "CUDA_Outputs/laplacian_of_gaussian_constant_shared.jpg";


    char* laplacian = "LAPLACIAN";
    char* gaussian = "GAUSSIAN";
    char* laplacian_of_gaussian = "LAPLACIAN OF GAUSSIAN";

    if (compareGrayscales(serial_grayscale, parallel_grayscale))
    {
        return 1;
    }

    if (compareImages(serial_laplacian, no_opts_laplacian, const_laplacian, const_shared_laplacian, laplacian))
    {
        return 1;
    }

    if (compareImages(serial_gaussian, no_opts_gaussian, const_gaussian, const_shared_gaussian, gaussian))
    {
        return 1;
    }

    if (compareImages(serial_laplacian_of_gaussian, no_opts_laplacian_of_gaussian, const_laplacian_of_gaussian, const_shared_laplacian_of_gaussian, laplacian_of_gaussian))
    {
        return 1;
    }

    return 0;
}


void serialSeparateFilters(stbi_uc* grayscale_image, int width, int height, int channels)
{
    printf("*****SERIAL SEPARATE FILTERS*****\n");

    float elapsed_time;
    cudaEvent_t serial_event_start, serial_event_stop;
    cudaEventCreate(&serial_event_start);
    cudaEventCreate(&serial_event_stop);
    cudaEventRecord(serial_event_start, 0);

    stbi_uc* laplacian_image = laplacian(grayscale_image, width, height, channels);
    stbi_uc* gaussian_image = gaussian(laplacian_image, width, height, channels);

    cudaEventRecord(serial_event_stop, 0);
    cudaEventSynchronize(serial_event_stop);
    cudaEventElapsedTime(&elapsed_time, serial_event_start, serial_event_stop);
    cudaEventDestroy(serial_event_start);
    cudaEventDestroy(serial_event_stop);
    
    printf("TIME:: %f\n", elapsed_time);

    const char* laplacian_output = "Outputs/laplacian.jpg";
    const char* gaussian_output = "Outputs/gaussian.jpg";
    writeImage(laplacian_output, laplacian_image, width, height, channels);
    writeImage(gaussian_output, gaussian_image, width, height, channels);

    imageFree(laplacian_image);
    imageFree(gaussian_image);
}

void serialCombinedFilter(stbi_uc* grayscale_image, int width, int height, int channels)
{
    printf("\n*****SERIAL COMBINED FILTER*****\n");

    float elapsed_time;
    cudaEvent_t serial_event_start, serial_event_stop;
    cudaEventCreate(&serial_event_start);
    cudaEventCreate(&serial_event_stop);
    cudaEventRecord(serial_event_start, 0);

    stbi_uc* laplacian_gaussian_image = laplacianGaussian(grayscale_image, width, height, channels);

    cudaEventRecord(serial_event_stop, 0);
    cudaEventSynchronize(serial_event_stop);
    cudaEventElapsedTime(&elapsed_time, serial_event_start, serial_event_stop);
    cudaEventDestroy(serial_event_start);
    cudaEventDestroy(serial_event_stop);
    
    printf("TIME:: %f\n", elapsed_time);

    const char* laplacian_gaussian_output = "Outputs/laplacian_of_gaussian.jpg";
    writeImage(laplacian_gaussian_output, laplacian_gaussian_image, width, height, channels);

    imageFree(laplacian_gaussian_image);
}


int serialImplementation(const char* path_to_input_image)
{
    int width, height, channels;
    stbi_uc* input_image = loadImage(path_to_input_image, &width, &height, &channels);
    if (input_image == NULL) {
        printf("Could not load image %s.\n", path_to_input_image);
        return 1;
    }

    // *****SERIAL IMPLEMENTATION*****
    stbi_uc* grayscale_image = grayscale(input_image, width, height, channels);

    serialSeparateFilters(grayscale_image, width, height, channels);
    serialCombinedFilter(grayscale_image, width, height, channels);

    // *****SERIAL OUTPUT IMAGES*****
    const char* grayscale_output = "Outputs/grayscale.jpg";
    writeImage(grayscale_output, grayscale_image, width, height, channels);

    // Free serial filter images
    imageFree(input_image);
    imageFree(grayscale_image);

    return 0;
}

void parallelSeparateFilters(stbi_uc* grayscale_cuda_image, int width, int height, int channels)
{
    printf("\n*****PARALLEL SEPARATE FILTERS*****\n");

    float elapsed_no_opts_time;
    float elapsed_const_mem_time;
    float elapsed_const_shared_mem_time;

    cudaEvent_t no_optimizations_start, no_optimizations_stop;
    cudaEvent_t constant_memory_start, constant_memory_stop;
    cudaEvent_t constant_shared_memory_start, constant_shared_memory_stop;

    // No Optimizations
    cudaEventCreate(&no_optimizations_start);
    cudaEventCreate(&no_optimizations_stop);
    cudaEventRecord(no_optimizations_start, 0);

    // *****************************************************************************************************************************
    stbi_uc* laplacian_no_opts_image = laplacianCudaNoOptimizations(grayscale_cuda_image, width, height, channels);
    stbi_uc* gaussian_no_opts_image = gaussianCudaNoOptimizations(laplacian_no_opts_image, width, height, channels);
    // *****************************************************************************************************************************

    cudaEventRecord(no_optimizations_stop, 0);
    cudaEventSynchronize(no_optimizations_stop);
    cudaEventElapsedTime(&elapsed_no_opts_time, no_optimizations_start, no_optimizations_stop);
    cudaEventDestroy(no_optimizations_start);
    cudaEventDestroy(no_optimizations_stop);
    printf("NO OPTIMIZATIONS TIME:: %f\n", elapsed_no_opts_time);


    // Constant Memory Usage
    cudaEventCreate(&constant_memory_start);
    cudaEventCreate(&constant_memory_stop);
    cudaEventRecord(constant_memory_start, 0);

    // *****************************************************************************************************************************
    stbi_uc* laplacian_const_image = laplacianCudaConstant(grayscale_cuda_image, width, height, channels);
    stbi_uc* gaussian_const_image = gaussianCudaConstant(laplacian_const_image, width, height, channels);
    // *****************************************************************************************************************************

    cudaEventRecord(constant_memory_stop, 0);
    cudaEventSynchronize(constant_memory_stop);
    cudaEventElapsedTime(&elapsed_const_mem_time, constant_memory_start, constant_memory_stop);
    cudaEventDestroy(constant_memory_start);
    cudaEventDestroy(constant_memory_stop);
    printf("CONSTANT MEMORY TIME:: %f\n", elapsed_const_mem_time);


    // Constant & Shared Memory Usage
    cudaEventCreate(&constant_shared_memory_start);
    cudaEventCreate(&constant_shared_memory_stop);
    cudaEventRecord(constant_shared_memory_start, 0);

    // *****************************************************************************************************************************
    stbi_uc* laplacian_const_shared_image = laplacianCudaConstantShared(grayscale_cuda_image, width, height, channels);
    stbi_uc* gaussian_const_shared_image = gaussianCudaConstantShared(laplacian_const_shared_image, width, height, channels);
    // *****************************************************************************************************************************

    cudaEventRecord(constant_shared_memory_stop, 0);
    cudaEventSynchronize(constant_shared_memory_stop);
    cudaEventElapsedTime(&elapsed_const_shared_mem_time, constant_shared_memory_start, constant_shared_memory_stop);
    cudaEventDestroy(constant_shared_memory_start);
    cudaEventDestroy(constant_shared_memory_stop);
    printf("CONSTANT & SHARED MEMORY TIME:: %f\n", elapsed_const_shared_mem_time);

    const char* laplacian_no_opts_output = "CUDA_Outputs/laplacian_no_optimizations.jpg";
    const char* laplacian_const_output = "CUDA_Outputs/laplacian_constant.jpg";
    const char* laplacian_const_shared_output = "CUDA_Outputs/laplacian_constant_shared.jpg";
    const char* gaussian_no_opts_output = "CUDA_Outputs/gaussian_no_optimizations.jpg";
    const char* gaussian_const_output = "CUDA_Outputs/gaussian_constant.jpg";
    const char* gaussian_const_shared_output = "CUDA_Outputs/gaussian_constant_shared.jpg";

    writeImage(laplacian_no_opts_output, laplacian_no_opts_image, width, height, channels);
    writeImage(laplacian_const_output, laplacian_const_image, width, height, channels);
    writeImage(laplacian_const_shared_output, laplacian_const_shared_image, width, height, channels);
    writeImage(gaussian_no_opts_output, gaussian_no_opts_image, width, height, channels);
    writeImage(gaussian_const_output, gaussian_const_image, width, height, channels);
    writeImage(gaussian_const_shared_output, gaussian_const_shared_image, width, height, channels);

    imageFree(laplacian_no_opts_image);
    imageFree(laplacian_const_image);
    imageFree(laplacian_const_shared_image);
    imageFree(gaussian_no_opts_image);
    imageFree(gaussian_const_image);
    imageFree(gaussian_const_shared_image);
}

void parallelCombinedFilter(stbi_uc* grayscale_cuda_image, int width, int height, int channels)
{
    printf("\n*****PARALLEL COMBINED FILTER*****\n");

    float elapsed_no_opts_time;
    float elapsed_const_mem_time;
    float elapsed_const_shared_mem_time;

    cudaEvent_t no_optimizations_start, no_optimizations_stop;
    cudaEvent_t constant_memory_start, constant_memory_stop;
    cudaEvent_t constant_shared_memory_start, constant_shared_memory_stop;

    // No Optimizations
    cudaEventCreate(&no_optimizations_start);
    cudaEventCreate(&no_optimizations_stop);
    cudaEventRecord(no_optimizations_start, 0);

    // *****************************************************************************************************************************
    stbi_uc* laplacian_gaussian_no_opts_image = laplacianGaussianCudaNoOptimizations(grayscale_cuda_image, width, height, channels);
    // *****************************************************************************************************************************

    cudaEventRecord(no_optimizations_stop, 0);
    cudaEventSynchronize(no_optimizations_stop);
    cudaEventElapsedTime(&elapsed_no_opts_time, no_optimizations_start, no_optimizations_stop);
    cudaEventDestroy(no_optimizations_start);
    cudaEventDestroy(no_optimizations_stop);
    printf("NO OPTIMIZATIONS TIME:: %f\n", elapsed_no_opts_time);


    // Constant Memory Usage
    cudaEventCreate(&constant_memory_start);
    cudaEventCreate(&constant_memory_stop);
    cudaEventRecord(constant_memory_start, 0);

    // *****************************************************************************************************************************
    stbi_uc* laplacian_gaussian_const_image = laplacianGaussianCudaConstant(grayscale_cuda_image, width, height, channels);
    // *****************************************************************************************************************************

    cudaEventRecord(constant_memory_stop, 0);
    cudaEventSynchronize(constant_memory_stop);
    cudaEventElapsedTime(&elapsed_const_mem_time, constant_memory_start, constant_memory_stop);
    cudaEventDestroy(constant_memory_start);
    cudaEventDestroy(constant_memory_stop);
    printf("CONSTANT MEMORY TIME:: %f\n", elapsed_const_mem_time);


    // Constant & Shared Memory Usage
    cudaEventCreate(&constant_shared_memory_start);
    cudaEventCreate(&constant_shared_memory_stop);
    cudaEventRecord(constant_shared_memory_start, 0);

    // *****************************************************************************************************************************
    stbi_uc* laplacian_gaussian_const_shared_image = laplacianGaussianCudaConstantShared(grayscale_cuda_image, width, height, channels);
    // *****************************************************************************************************************************

    cudaEventRecord(constant_shared_memory_stop, 0);
    cudaEventSynchronize(constant_shared_memory_stop);
    cudaEventElapsedTime(&elapsed_const_shared_mem_time, constant_shared_memory_start, constant_shared_memory_stop);
    cudaEventDestroy(constant_shared_memory_start);
    cudaEventDestroy(constant_shared_memory_stop);
    printf("CONSTANT & SHARED MEMORY TIME:: %f\n", elapsed_const_shared_mem_time);

    // *****PARALLEL OUTPUT IMAGES*****
    const char* laplacian_gaussian_no_opts_output = "CUDA_Outputs/laplacian_of_gaussian_no_optimizations.jpg";
    const char* laplacian_gaussian_const_output = "CUDA_Outputs/laplacian_of_gaussian_constant.jpg";
    const char* laplacian_gaussian_const_shared_output = "CUDA_Outputs/laplacian_of_gaussian_constant_shared.jpg";

    writeImage(laplacian_gaussian_no_opts_output, laplacian_gaussian_no_opts_image, width, height, channels);
    writeImage(laplacian_gaussian_const_output, laplacian_gaussian_const_image, width, height, channels);
    writeImage(laplacian_gaussian_const_shared_output, laplacian_gaussian_const_shared_image, width, height, channels);

    imageFree(laplacian_gaussian_no_opts_image);
    imageFree(laplacian_gaussian_const_image);
    imageFree(laplacian_gaussian_const_shared_image);
}


int parallelImplementation(const char* path_to_input_image)
{
    int width, height, channels;
    stbi_uc* cuda_input_image = loadImage(path_to_input_image, &width, &height, &channels);
    if (cuda_input_image == NULL)
    {
        printf("Could not load image %s.\n", path_to_input_image);
        return 1;
    }

    // *****PARALLEL IMPLEMENTATION*****
    stbi_uc* grayscale_cuda_image = grayscaleCuda(cuda_input_image, width, height, channels);

    parallelSeparateFilters(grayscale_cuda_image, width, height, channels);
    parallelCombinedFilter(grayscale_cuda_image, width, height, channels);

    const char* grayscale_cuda_output = "CUDA_Outputs/grayscale_cuda.jpg";
    writeImage(grayscale_cuda_output, grayscale_cuda_image, width, height, channels);


    // // Free parallel filter images
    imageFree(cuda_input_image);
    imageFree(grayscale_cuda_image);
    return 0;
}

int main(int argc, const char* argv[])
{
    const char* path_to_input_image = argv[1];

    if (serialImplementation(path_to_input_image))
    {
        return 1;
    }
    if (parallelImplementation(path_to_input_image))
    {
        return 1;
    }
    
    printf("\n\n");
    performValidation();

    return 0;
}