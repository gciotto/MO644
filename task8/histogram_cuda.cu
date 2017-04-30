/*
  Task #8 - Gustavo Ciotto Pinton
  MO644 - Parallel Programming
*/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255

#include <cuda.h>

#define THREAD_PER_BLOCK 1024 /* Tesla k40 supports 1024 threads per block */

typedef struct {
	unsigned char red, green, blue;
} PPMPixel;

typedef struct {
	int x, y;
	PPMPixel *data;
} PPMImage;

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


static PPMImage *readPPM(const char *filename) {
	char buff[16];
	PPMImage *img;
	FILE *fp;
	int c, rgb_comp_color;
	fp = fopen(filename, "rb");
	if (!fp) {
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	if (!fgets(buff, sizeof(buff), fp)) {
		perror(filename);
		exit(1);
	}

	if (buff[0] != 'P' || buff[1] != '6') {
		fprintf(stderr, "Invalid image format (must be 'P6')\n");
		exit(1);
	}

	img = (PPMImage *) malloc(sizeof(PPMImage));
	if (!img) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	c = getc(fp);
	while (c == '#') {
		while (getc(fp) != '\n')
			;
		c = getc(fp);
	}

	ungetc(c, fp);
	if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
		fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
		exit(1);
	}

	if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
		fprintf(stderr, "Invalid rgb component (error loading '%s')\n",
				filename);
		exit(1);
	}

	if (rgb_comp_color != RGB_COMPONENT_COLOR) {
		fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
		exit(1);
	}

	while (fgetc(fp) != '\n')
		;
	img->data = (PPMPixel*) malloc(img->x * img->y * sizeof(PPMPixel));

	if (!img) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
		fprintf(stderr, "Error loading image '%s'\n", filename);
		exit(1);
	}

	fclose(fp);
	return img;
}

__global__ void cudaHistogram (PPMPixel *data, int size, int *h) {

	int i = threadIdx.x + blockIdx.x * blockDim.x,
		stride = blockDim.x * gridDim.x; /* Gives the number of threads in a grid */ 

	while (i < size) {

		/* Implicit conversion from float to int gives the same result of floor() function */
		int r = ( (float) (data[i].red * 4) / 256),
		    g = ( (float) (data[i].green * 4) / 256),
		    b = ( (float) (data[i].blue * 4) / 256);

		int x = r * 16 + g * 4 + b;

		atomicAdd(&h[x], 1);

		i += stride;
	}

}

int main(int argc, char *argv[]) {

	if( argc != 2 ) {
		printf("Too many or no one arguments supplied.\n");
		return 0;
	}

	int i, n;
	char *filename = argv[1]; //Recebendo o arquivo!;
	double start, end;
	
	//scanf("%s", filename);
	PPMImage *image = readPPM(filename);
	n = image->x * image->y;
	
        int *h = (int*)malloc(sizeof(int) * 64);

	/* We consider in the execution delay the memory allocation time */
	start = rtclock();

	/* Allocating memory for image data in the device */
	int image_size = n * sizeof(PPMPixel);
	PPMPixel *cuda_image;
	cudaMalloc((void**) &cuda_image, image_size);
	cudaMemcpy(cuda_image, image->data, image_size, cudaMemcpyHostToDevice);

	/* Allocating memory for histogram in the device */
	int *cuda_h;
	cudaMalloc((void**) &cuda_h, 64 * sizeof(int));
	cudaMemset(cuda_h, 0, 64 * sizeof(int));

	/* Computes how many blocks will be used. */
	int cuda_blocks = ceil ( (float) n / THREAD_PER_BLOCK );

	cudaHistogram <<< cuda_blocks, THREAD_PER_BLOCK >>> (cuda_image, n, cuda_h);

	/* Copying computed result from device memory */
	cudaMemcpy(h, cuda_h, sizeof(int) * 64, cudaMemcpyDeviceToHost);

	/* As cudaMemcpy is a blocking call, we do not need to call cudaThreadSynchronize() */
	end = rtclock();

	for (i = 0; i < 64; i++){
		printf("%0.3f ", (float) h[i] / n);
	}
	printf("\n");

	printf("\n%0.6lfs\n", end - start);
	
	/* Cleaning everything up */
	free(h);
	free(image->data);
	free(image);

	cudaFree(cuda_image);
	cudaFree(cuda_h);
}
