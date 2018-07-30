#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "main_functions.h"
#include <device_functions.h>
#include <thrust\device_vector.h>

#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

__global__ void calc2points(float* point_coordinate_1, float* point_coordinate_2 , float* coordinates_arr);
cudaError_t calcDistanceCoordiantesWithCuda(float* coordinates_1, float* coordinates_2, float* coordinates_arr, int num_coordinates);

void print_vars(int N, int n,int MAX,int LIMIT, float QM);
void error(void* void_1 , void* void_2, void* void_3);

int main()
{
	clock_t begin = clock();

	int k = 2; // init number of cluster
	int N,n,MAX,LIMIT; // 250, 52, 30, 200
	float QM; // 7.3
	float* arr_var = (float*)malloc(sizeof(float)*5);

	arr_var = getVarsFromTXT();
	N = (int)arr_var[0]; // Number of point
	n = (int)arr_var[1]; // Number of coordinates
	MAX = (int)arr_var[2]; 
	LIMIT = (int)arr_var[3]; // LIMIT iterations
	QM = arr_var[4]; 
	print_vars(N,n,MAX,LIMIT,QM);
	Point* all_points = (Point*)malloc(sizeof(Point)*N);
	all_points = getPointsFromTXT(all_points,N,n); // get Points
	Cluster* all_clusters = (Cluster*)calloc(k, sizeof(Cluster)); // Create Clusters

	//print_number_of_points_cluster(all_clusters, k);
	//print_diamters(all_clusters,k);

	all_clusters = init_clusters(all_clusters,all_points, k, N); // initiate clusters

	printf("Iterations Starts:\n----------------------\n");

	while(k < MAX) {

		printf("Number of Clusters = %d\n",k);
		for (int i = 0 ; i < LIMIT ; i++)
		{
			printf("\nIteration %d:\n------------------\n",i);

			associate_points_to_clusters(all_points, all_clusters, N, k, n); // Associate points to clusters
			print_cluster_ditails(all_clusters, k);
			printf("Recenter clusters ..\n");
			recenter_all_clusters(all_clusters, k , n); // Recenter Clusters
			
			//print_clusters_centers(all_clusters, k , n);
			//system("pause");
			if(check_termination_condition(all_clusters, k, n))
			{
				printf("Termination condition Happened\n\n");
				i = LIMIT;
			}
			else
			{
				reset_points_and_diameter_in_clusters(all_clusters, k);
				//associate_points_to_clusters(all_points, all_clusters, N, k, n); // Associate points to clusters checking
			}
		}

		double new_quality = calc_qm(all_clusters,k, n);
		//double new_quality = calculateQM(all_clusters,k,n);

		printf("new_quality:%f\n",new_quality);

		if(new_quality <= QM)
		{	
			clock_t end = clock();
			double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

			printf("\nFinished with good quality = %f\n", new_quality);
			printf("Number of clusters: %d\n",k);

			print_cluster_ditails(all_clusters, k);

			if (parallel)
				printf("In Parallel:\n");
			else
				printf("In Sequence:\n");
			printf("Time Spent: %f Seconds\n",time_spent);

			write_results_to_file(all_clusters, k, n, new_quality);

			free(all_points);
			free(all_clusters);
			break;
		}
		else 
		{
			printf("The new QM is bigger than %.2f\n",QM);
			printf("Increase the number of clusters\n\n");
			k++;
			free(all_clusters);
			all_clusters = (Cluster*)calloc(k, sizeof(Cluster));
			all_clusters = init_clusters(all_clusters,all_points, k, N); // initiate clusters
		}
	}
}

void print_vars(int N, int n,int MAX,int LIMIT, float QM)
{
	printf("Variables:\n------------\n");
	printf("N = %d Products\n",N);
	printf("n = %d Coordinates\n",n);
	printf("MAX = %d Maximum Clusters\n",MAX);
	printf("LIMIT = %d Limit Iterations\n",LIMIT);
	printf("QM = %.2f Quality Clusters\n\n",QM);
}

void error(void* void_1 , void* void_2, void* void_3)
{
	cudaFree(void_1);
	cudaFree(void_2);
	cudaFree(void_3);
}


__global__ void calc2points(float* point_coordinate_1, float* point_coordinate_2 , float* coordinates_arr)
{
    int tid = threadIdx.x; // 52

	coordinates_arr[tid] = pow(point_coordinate_1[tid] - point_coordinate_2[tid],2);
}


 //Helper function for using CUDA to add vectors in parallel.
cudaError_t calcDistanceCoordiantesWithCuda(float* coordinates_1, float* coordinates_2, float* coordinates_arr, int num_coordinates)
{
    float* dev_coordinates_1;
    float* dev_coordinates_2;
	float* dev_coordinates_arr;

	dev_coordinates_1 = (float*)malloc(sizeof(float)*num_coordinates);
	dev_coordinates_2 = (float*)malloc(sizeof(float)*num_coordinates);
	dev_coordinates_arr = (float*)malloc(sizeof(float)*num_coordinates);
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        //goto Error;
		error(dev_coordinates_1, dev_coordinates_2, dev_coordinates_arr);
    }

    // Allocate GPU buffers for three vectors (two input, one output)
	// dev_coordinates_1
	cudaStatus = cudaMalloc((void**)&dev_coordinates_1, sizeof(float)*num_coordinates);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc of dev_coordinates_1 failed!\n");
        //goto Error;
		error(dev_coordinates_1, dev_coordinates_2, dev_coordinates_arr);
    }

	// dev_coordinates_2
    cudaStatus = cudaMalloc((void**)&dev_coordinates_2, sizeof(float)*num_coordinates);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc of dev_coordinates_2 failed!\n");
        //goto Error;
		error(dev_coordinates_1, dev_coordinates_2, dev_coordinates_arr);
    }

	// coordinates_arr
	cudaStatus = cudaMalloc((void**)&dev_coordinates_arr, sizeof(float)*num_coordinates);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc of dev_coordinates_arr failed!\n");
        //goto Error;
		error(dev_coordinates_1, dev_coordinates_2, dev_coordinates_arr);
    }

	
    // Copy from host memory to GPU buffers.
	cudaStatus = cudaMemcpyAsync(dev_coordinates_1, coordinates_1, sizeof(float)*num_coordinates, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyAsync dev_coordinates_1 failed!");
        //goto Error;
		error(dev_coordinates_1, dev_coordinates_2, dev_coordinates_arr);
    }

	cudaStatus = cudaMemcpyAsync(dev_coordinates_2, coordinates_2, sizeof(float)*num_coordinates, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyAsync dev_coordinates_2 failed!");
        //goto Error;
		error(dev_coordinates_1, dev_coordinates_2, dev_coordinates_arr);
    }

	cudaStatus = cudaMemcpyAsync(dev_coordinates_arr, coordinates_arr, sizeof(float)*num_coordinates, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyAsync dev_coordinates_arr failed!");
		printf("stderr:%s\n",stderr);
        //goto Error;
		error(dev_coordinates_1, dev_coordinates_2, dev_coordinates_arr);
    }

    // Launch a kernel on the GPU with one thread for each element.
	calc2points<<<1, 52>>>(dev_coordinates_1, dev_coordinates_2 ,dev_coordinates_arr);
	//calc2pointsWith4Blocks<<<4, 13>>>(dev_coordinates_1, dev_coordinates_2 ,dev_coordinates_arr);
    // Check for any errors launching the kernel
    
	cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //goto Error;
		error(dev_coordinates_1, dev_coordinates_2, dev_coordinates_arr);
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        //goto Error;
		error(dev_coordinates_1, dev_coordinates_2, dev_coordinates_arr);
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpyAsync(coordinates_arr, dev_coordinates_arr, sizeof(float)*num_coordinates, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        //goto Error;
		error(dev_coordinates_1, dev_coordinates_2, dev_coordinates_arr);
    }
	error(dev_coordinates_1, dev_coordinates_2, dev_coordinates_arr);
	return cudaStatus;
}