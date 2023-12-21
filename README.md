# Laplacian of Gaussian Blob Detection Project
## By: Conner DeJong

The Laplacian of Guassian project allows a user to execute the program 
with the input file of their choice, and will output a series of images
which have had filters applied in the following orders:
- Grayscale Filter
- Laplacian Filter
- Gaussian Smoothing
OR
- Grayscale Filter
- Laplacian of Gaussian Filter

The following steps are required to compile and execute the program:
1. Open the terminal to the directory containing the project files
2. Enter the command "make", which will compile the files into an 
executable
3. Enter the command: "./image **path_to_input_image**", and the filters 
will be applied to the specified image
4. Wait for program to finish execution (this will take time for the
serial and parallel implementations to finish)
5. Observe timing results in the terminal, and output image results in
the output directories "Outputs" and "CUDA-Outputs"

Sample input images are located in the subdirectory "Inputs" for the user
to test the program with. An example execution of the program (after compilation)
would be "./image Inputs/flower.jpg".