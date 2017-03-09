#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

int main (){

	int n_threads, size_array;
	float *unsorted_array, *sorted_array;

	scanf("%d", &n_threads);
	scanf("%d", &size_array);

	unsorted_array = (float*) malloc(size_array * sizeof(float));
	sorted_array = (float*) malloc(size_array * sizeof(float));

	for (int i = 0; i < size_array; i++)
		scanf("%f", unsorted_array + i);

	/* parallelizable code */

	double start = omp_get_wtime(), end;
# 	pragma omp parallel for num_threads(n_threads)
	for (int i = 0; i < size_array; i++) {
		int index = 0;

		for (int j = 0; j < size_array; j++)
			if (i != j) {
				if (unsorted_array[i] > unsorted_array[j])
					index++;
				else if (unsorted_array[i] == unsorted_array[j] && i > j)
					index++;
			}

		sorted_array[index] = unsorted_array[i];
	}
	/* end of parallelizable code */
	end = omp_get_wtime() - start;

	for (int i = 0; i < size_array; i++)
		printf("%.2f ", sorted_array[i]);

	printf("\n%f\n", end);

	free(unsorted_array);
	free(sorted_array);

	return 0;

}
