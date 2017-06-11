/*
 * Task #9 - Parallel Programming
 * 
 * Gustavo Ciotto Pinton
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255

#include <cuda.h>

#define THREAD_PER_BLOCK 1024 /* Tesla k40 supports 1024 threads per block */

/* Mask attributes  */
#define MASK_WIDTH 5
#define RADIUS (MASK_WIDTH-1)/2

#define PIXEL(R,G,B) ( (PPMPixel) { .red = (R), .green = (G), .blue = (B)})

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

void writePPM(PPMImage *img) {

    fprintf(stdout, "P6\n");
    fprintf(stdout, "# %s\n", COMMENT);
    fprintf(stdout, "%d %d\n", img->x, img->y);
    fprintf(stdout, "%d\n", RGB_COMPONENT_COLOR);

    fwrite(img->data, 3 * img->x, img->y, stdout);
    fclose(stdout);
}

/* kernel function. It loads MASK_WIDTH rows containing the same number of elements as the number of threads per block +2 
 into the shared memory and computes the new value for the pixel based on this data. */
__global__ void cudaSmoothing (PPMPixel *data_in, PPMPixel *data_out, int columns, int rows) {

	__shared__ PPMPixel shared_data [MASK_WIDTH][THREAD_PER_BLOCK + 2 * RADIUS];

	int i = threadIdx.x + blockIdx.x * blockDim.x,
		shared_i = threadIdx.x + RADIUS,
		n = rows * columns, j, k,
		total_red, total_blue, total_green;
	

	if ( i  < n ) {

		/* Populating shared memory */
		for (j = 0; j < MASK_WIDTH; j++) {

			/* Computes the index of the array corresponding to the column in row (j - RADIUS) */
			int col_index = i + (j - RADIUS) * columns;

			shared_data [j][shared_i] = PIXEL(0,0,0);

			/* only if i is inside the image border */
			if (col_index >= 0 && col_index < n)
				shared_data [j][shared_i] = data_in [col_index];

			if (threadIdx.x < RADIUS) {

				shared_data [j][shared_i - RADIUS] = PIXEL(0,0,0);
				shared_data [j][shared_i + THREAD_PER_BLOCK] = PIXEL(0,0,0);

				if ((col_index - RADIUS) >= 0)
					shared_data [j][shared_i - RADIUS] = data_in [col_index - RADIUS];

				if ((col_index + THREAD_PER_BLOCK) < n) 
					shared_data [j][shared_i + THREAD_PER_BLOCK] = data_in [col_index + THREAD_PER_BLOCK];
			}

		}

		/* Ensures all threads updated the shared memory */
		__syncthreads();

		total_red = total_blue = total_green = 0;

		int left_border = i - (i % columns),
			right_border = i + columns - (i % columns);

		/* Iterates over lines */
		for (j = 0; j < MASK_WIDTH; j++) {
			
			/* Iterates over columns */
			for (k = - RADIUS; k <= RADIUS; k++) {

				/* We must check if the current element is not in the border. In this case, we need to avoid
				   summing the other side element */
				if (i + k >= left_border && i + k < right_border) {

					total_red += shared_data[j][shared_i +  k].red;
					total_blue += shared_data[j][shared_i + k].blue;
					total_green += shared_data[j][shared_i + k].green;
				}
			}

		}

		data_out[i].red = total_red / ( MASK_WIDTH * MASK_WIDTH );
		data_out[i].blue = total_blue / ( MASK_WIDTH * MASK_WIDTH );
		data_out[i].green = total_green / ( MASK_WIDTH * MASK_WIDTH );
	}

}

int main(int argc, char *argv[]) {

    if( argc != 2 ) {
        printf("Too many or no one arguments supplied.\n");
    }

#ifdef PRINT_TIME
    double t_start, t_end;
#endif

    char *filename = argv[1]; //Recebendo o arquivo!;
    PPMImage *image = readPPM(filename);
    PPMImage *image_output = readPPM(filename);

    /* Number of elements in the image */
    int n = image->x * image->y;

#ifdef PRINT_TIME
    t_start = rtclock();
#endif

    /* Allocating memory for image data in the device */
    int image_size = n * sizeof(PPMPixel);
    PPMPixel *cuda_image_data;
    cudaMalloc((void**) &cuda_image_data, image_size);

    /* Copying image data to the device */
    cudaMemcpy(cuda_image_data, image->data, image_size, cudaMemcpyHostToDevice);

    /* Allocating memory for image result in the device */
    PPMPixel *cuda_image_out;
    cudaMalloc((void**) &cuda_image_out, image_size);

    /* Computes how many blocks will be used. */
    int cuda_blocks = ceil ( (float) n / THREAD_PER_BLOCK );

    cudaSmoothing <<< cuda_blocks, THREAD_PER_BLOCK >>> (cuda_image_data, cuda_image_out, image->x, image->y);

    /* Copying computed result from device memory */
    cudaMemcpy(image_output->data, cuda_image_out, image_size, cudaMemcpyDeviceToHost);

#ifdef PRINT_TIME
    t_end = rtclock();
	fprintf(stdout, "\n%0.6lfs\n", t_end - t_start);
#else
    writePPM(image_output);
#endif

    free(image->data);
    free(image);
    free(image_output->data);
    free(image_output);

    cudaFree(cuda_image_data);
    cudaFree(cuda_image_out);
}
