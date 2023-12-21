#ifndef IMAGE_H__
#define IMAGE_H__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
extern "C"
// Needed for reading images
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"

// Needed for writing images
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

#define RGBA    4
#define PI      3.14159265

#define GRAYSCALE_BLOCK_SIZE    32

#define LAPLACIAN_KERNEL_SIZE   3
#define LAPLACIAN_TILE_SIZE     14
#define LAPLACIAN_BLOCK_SIZE    (LAPLACIAN_TILE_SIZE + LAPLACIAN_KERNEL_SIZE - 1)

#define GAUSSIAN_KERNEL_SIZE    5
#define GAUSSIAN_TILE_SIZE      12
#define GAUSSIAN_BLOCK_SIZE     (GAUSSIAN_TILE_SIZE + GAUSSIAN_KERNEL_SIZE - 1)

#define LOG_KERNEL_SIZE         9
#define LOG_TILE_SIZE           23
#define LOG_BLOCK_SIZE          (LOG_TILE_SIZE + LOG_KERNEL_SIZE - 1)

typedef struct Pixel {
    stbi_uc red;
    stbi_uc green;
    stbi_uc blue;
} Pixel;

stbi_uc* loadImage(const char* path_to_image, int* width, int* height, int* channels) {
    return stbi_load(path_to_image, width, height, channels, STBI_rgb);
}

void writeImage(const char* path_to_image, stbi_uc* image, int width, int height, int channels) {
    stbi_write_jpg(path_to_image, width, height, channels, image, 95);
}

void imageFree(stbi_uc* image) {
    stbi_image_free(image);
}

void getPixel(stbi_uc* image, int width, int height, int channels, int x, int y, Pixel *ret_pixel)
{
    stbi_uc* pixel = &image[channels * (y*width + x)];
    ret_pixel->red = pixel[0];
    ret_pixel->green = pixel[1];
    ret_pixel->blue = pixel[2];
}

void setPixel(stbi_uc *image, int width, int height, int channels, int x, int y, Pixel *output_pixel)
{
    unsigned char* set_pixel = &image[channels * (y*width + x)];
    set_pixel[0] = output_pixel->red;
    set_pixel[1] = output_pixel->green;
    set_pixel[2] = output_pixel->blue;
}

void resetPixel(Pixel *pixel)
{
    pixel->red = 0;
    pixel->green = 0;
    pixel->blue = 0;
}

float relativeLuminance(Pixel *pixel)
{
    // L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    float red_value = (float)pixel->red * 0.2126;
    float green_value = (float)pixel->green * 0.7152;
    float blue_value = (float)pixel->blue * 0.0722;
    return ((red_value) + (green_value) + (blue_value));
}

stbi_uc* grayscale(stbi_uc* input, int width, int height, int channels)
{
    int image_size = width * height * channels;
    stbi_uc* output = (stbi_uc*) malloc(sizeof(stbi_uc)*image_size);
    Pixel output_pixel;
    Pixel current_pixel;
    float luminance = 0.0f;

    for (int x=0; x<width; x++)
    {
        for (int y=0; y<height; y++)
        {
            resetPixel(&output_pixel);
            resetPixel(&current_pixel);
            getPixel(input, width, height, channels, x, y, &current_pixel);
            luminance = relativeLuminance(&current_pixel);
            output_pixel.red = (unsigned char) (luminance);
            output_pixel.green = (unsigned char) (luminance);
            output_pixel.blue = (unsigned char) (luminance);
            setPixel(output, width, height, channels, x, y, &output_pixel);
        }
    }

    return output;
}

int getLaplacianCoordinate(int inner, int outer)
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

stbi_uc* laplacian(stbi_uc* input, int width, int height, int channels)
{
    int laplacianMatrix[3][3] = {{-1, -1, -1},
                                 {-1, 8, -1},
                                 {-1, -1, -1}};

    int image_size = width * height * channels;
    stbi_uc* output = (stbi_uc*) malloc(sizeof(stbi_uc)*image_size);
    Pixel output_pixel;
    Pixel current_pixel;
    float current_color_value;
    int index_x;
    int index_y;

    for (int outer_x=0; outer_x<width; outer_x++)
    {
        for (int outer_y=0; outer_y<height; outer_y++)
        {
            resetPixel(&output_pixel);
            resetPixel(&current_pixel);
            current_color_value = 0.0f;
            for (int inner_x=outer_x-1; inner_x<outer_x+2; inner_x++)
            {
                for (int inner_y=outer_y-1; inner_y<outer_y+2; inner_y++)
                {
                    if ((inner_x >= 0) && (inner_x < width) && (inner_y >= 0) && (inner_y < height))
                    {
                        getPixel(input, width, height, channels, inner_x, inner_y, &current_pixel);
                        index_x = getLaplacianCoordinate(inner_x, outer_x);
                        index_y = getLaplacianCoordinate(inner_y, outer_y);
                        current_color_value += ((relativeLuminance(&current_pixel)) * laplacianMatrix[index_x][index_y]);
                    }
                }
            }
            output_pixel.red = current_color_value;
            output_pixel.green = current_color_value;
            output_pixel.blue = current_color_value;
            setPixel(output, width, height, channels, outer_x, outer_y, &output_pixel);
        }
    }

    return output;
}

int getGaussianCoordinate(int inner, int outer)
{
    if (inner == outer-2)
    {
        return 0;
    }
    else if (inner == outer-1)
    {
        return 1;
    }
    else if (inner == outer)
    {
        return 2;
    }
    else if (inner == outer+1)
    {
        return 3;
    }
    else
    {
        return 4;
    }
}


stbi_uc* gaussian(stbi_uc* input, int width, int height, int channels)
{
    int gaussian_convolution[5][5] = {{1,  4,  6,  4,  1},
                                    {4, 16, 24, 16,  4},
                                    {6, 24, 36, 24,  6},
                                    {4, 16, 24, 16,  4},
                                    {1,  4,  6,  4,  1}};
    int image_size = width * height * channels;
    stbi_uc* output = (stbi_uc*) malloc(sizeof(stbi_uc)*image_size);
    Pixel output_pixel;
    Pixel current_pixel;
    int x_;
    int y_;
    float current_color_value;
    for (int outer_x=0; outer_x<width; outer_x++)
    {
        for (int outer_y=0; outer_y<height; outer_y++)
        {
            resetPixel(&current_pixel);
            resetPixel(&output_pixel);
            current_color_value = 0.0f;
            for (int inner_x=outer_x-2; inner_x < outer_x+3; inner_x++)
            {
                for (int inner_y=outer_y-2; inner_y < outer_y+3; inner_y++)
                {
                    if ((inner_x >= 0) && (inner_x < width) && 
                    (inner_y >= 0) && (inner_y < height))
                    {
                        getPixel(input, width, height, channels, inner_x, inner_y, &current_pixel);
                        x_ = getGaussianCoordinate(inner_x, outer_x);
                        y_ = getGaussianCoordinate(inner_y, outer_y);
                        current_color_value = current_color_value + (relativeLuminance(&current_pixel) * gaussian_convolution[x_][y_]);
                    }
                }
            }
            // divide by 256?
            output_pixel.red = (unsigned char)(current_color_value / 273);
            output_pixel.green = (unsigned char)(current_color_value / 273);
            output_pixel.blue = (unsigned char)(current_color_value / 273);
            setPixel(output, width, height, channels, outer_x, outer_y, &output_pixel);
        }
    }

    return output;
}


int getLoGCoordinate(int inner, int outer)
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

stbi_uc* laplacianGaussian(stbi_uc* input, int width, int height, int channels)
{
    int laplacian_gaussian_matrix[9][9] = {
                                        {0, 1, 1, 2, 2, 2, 1, 1, 0},
                                        {1, 2, 4, 5, 5, 5, 4, 2, 1},
                                        {1, 4, 5, 3, 0, 3, 5, 4, 1},
                                        {2, 5, 3, -12, -24, -12, 3, 5, 2},
                                        {2, 5, 0, -24, -40, -24, 0, 5, 2},
                                        {2, 5, 3, -12, -24, -12, 3, 5, 2},
                                        {1, 4, 5, 3, 0, 3, 5, 4, 1},
                                        {1, 2, 4, 5, 5, 5, 4, 2, 1},
                                        {0, 1, 1, 2, 2, 2, 1, 1, 0},
                                        };
    int image_size = width * height * channels;
    stbi_uc* output = (stbi_uc*) malloc(sizeof(stbi_uc)*image_size);
    Pixel output_pixel;
    Pixel current_pixel;
    float current_color_value;
    int log_x, log_y;
    for (int outer_x=0; outer_x<width; outer_x++)
    {
        for (int outer_y=0; outer_y<height; outer_y++)
        {
            resetPixel(&current_pixel);
            resetPixel(&output_pixel);
            current_color_value = 0.0f;
            for (int inner_x=outer_x-4; inner_x < outer_x+5; inner_x++)
            {
                for (int inner_y=outer_y-4; inner_y < outer_y+5; inner_y++)
                {
                    if ((inner_x >= 0) && (inner_x < width) && 
                    (inner_y >= 0) && (inner_y < height))
                    {
                        getPixel(input, width, height, channels, inner_x, inner_y, &current_pixel);
                        log_x = getLoGCoordinate(inner_x, outer_x);
                        log_y = getLoGCoordinate(inner_y, outer_y);
                        current_color_value = current_color_value + 
                        ((relativeLuminance(&current_pixel))*laplacian_gaussian_matrix[log_x][log_y]);
                    }
                }
            }

            output_pixel.red = (unsigned char)(current_color_value);
            output_pixel.green = (unsigned char)(current_color_value);
            output_pixel.blue = (unsigned char)(current_color_value);
            setPixel(output, width, height, channels, outer_x, outer_y, &output_pixel);
        }
    }

    return output;
}

#endif