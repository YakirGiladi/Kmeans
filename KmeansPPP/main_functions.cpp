#define _CRT_SECURE_NO_WARNINGS
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "main_functions.h"
#include "kernel.cuh"

void print_number_of_points_cluster(Cluster clusters);
void print_diamter(Cluster cluster);
void print_cluster_ditails(Cluster* all_clusters, int num_clusters);
void print_point_of_cluster(Cluster cluster, int num_coordinates);
void print_cluster_center(Cluster cluster, int num_coordinates);
void print_clusters_centers(Cluster* all_clusters, int num_clusters, int num_coordinates);
Cluster* init_clusters(Cluster* all_clusters, Point* all_points, int num_clusters , int num_points);
void reset_points_and_diameter_in_clusters(Cluster* all_clusters, int num_clusters);
float* getVarsFromTXT();
Point* getPointsFromTXT(Point* all_points,int N, int n);
double calc_distance_between_2_coordinate(double coor_1, double coor_2, int square);
double calc_distance_between_2_clusters(Cluster cluster_1, Cluster cluster_2, int num_coordinates);
double get_distance_between_2_points(Point point_1,Point point_2, int num_coordinates);
void associate_points_to_clusters(Point* all_points, Cluster* all_clusters, int num_points, int num_clusters, int num_coordinates);
void recenter_all_clusters(Cluster* all_clusters,int num_clusters, int num_coordinates);
double calc_diameter_of_single_cluster(Cluster* cluster, int num_coordinates);
void calc_diameter_of_all_clusters(Cluster* all_clusters,int num_clusters, int num_coordinates);
double calc_qm (Cluster* all_clusters,int num_clusters, int num_coordinates);
double sum_coordinates_arr(float* corrdinates, int num_coordinates);
float* get_diameter_of_all_clusters (Cluster* all_clusters, int num_clusters);
int* get_num_of_points_of_all_clusters (Cluster* all_clusters, int num_clusters);
int calc_min(float* dist_arr, int num_clusters);
bool check_termination_condition(Cluster* all_clusters, int num_clusters, int num_coordinates);
void print_logs(int var);
float* reset_arr(float* arr, int size);
void write_results_to_file(Cluster* all_clusters, int num_clusters, int num_coordinates, double qm);
void recenter_single_cluster(Cluster* cluster, int num_coordinates);


void print_number_of_points_cluster(Cluster clusters)
{
	printf("Points Size: %d\n",clusters.num_points_in_cluster);
}

void print_diamter(Cluster cluster)
{
	printf("Diamters: %.4f\n",cluster.diameter);
}

void print_cluster_ditails(Cluster* all_clusters, int num_clusters)
{
	for (int i = 0 ; i < num_clusters ; i++)
	{
		printf("Cluster %d:\n",i);
		print_number_of_points_cluster(all_clusters[i]);
		//print_diamter(all_clusters[i]);
		printf("\n");
	}
}

void print_point_of_cluster(Cluster cluster, int num_coordinates)
{
	for (int i = 0 ; i < cluster.num_points_in_cluster ; i++)
	{
		printf("\nPoint %d:\n",i);
		for (int j = 0 ; j < num_coordinates ; j++) 
		{
			printf("%.2f ",cluster.points_in_cluster[i].coordinates[j]);
		}
	}

}

void print_cluster_center(Cluster cluster, int num_coordinates)
{
	for (int i = 0 ; i < num_coordinates ; i++)
	{
		printf("%.3f	",cluster.center.coordinates[i]);
		if ((i+1) % 13 == 0)
			printf("\n");
	}
}

void print_clusters_centers(Cluster* all_clusters, int num_clusters, int num_coordinates)
{
	for (int i = 0 ; i < num_clusters ; i++) 
	{
		printf("\nCluster %d:\n", i);
		print_cluster_center(all_clusters[i], num_coordinates);
	}
}

Cluster* init_clusters(Cluster* all_clusters, Point* all_points, int num_clusters , int num_points) 
{
	if (parallel) // OMP
	{
		#pragma omp parallel for
		for (int i = 0 ; i < num_clusters ; i++)
		{
			all_clusters[i].center = all_points[i];
			all_clusters[i].num_points_in_cluster = 0;
			all_clusters[i].points_in_cluster = (Point*)calloc(num_points , sizeof(Point));
		}
	}
	else // Squence
	{
		for (int i = 0 ; i < num_clusters ; i++)
		{
			all_clusters[i].center = all_points[i];
			all_clusters[i].num_points_in_cluster = 0;
			all_clusters[i].points_in_cluster = (Point*)calloc(num_points , sizeof(Point));
		}
	}
	return all_clusters;
}

void reset_points_and_diameter_in_clusters(Cluster* all_clusters, int num_clusters) 
{
	if (parallel) // OMP
	{
		#pragma omp parallel for
		for (int i = 0; i < num_clusters; i++) 
		{
		   all_clusters[i].num_points_in_cluster = 0;
		   all_clusters[i].diameter = 0;
		}
	}
	else // Sequence
	{
		for (int i = 0; i < num_clusters; i++) 
		{
		   all_clusters[i].num_points_in_cluster = 0;
		   all_clusters[i].diameter = 0;
		}
	}
}

float* getVarsFromTXT()
{
    const char* frd = "C:/Users/yakir.giladi/Documents/Visual Studio 2010/Projects/KmeansPPP/Sales_Transactions_Dataset_Weekly.txt";

	FILE* f = fopen(frd, "r+");
	char str[10];
	float arr_var[5];
	
	fgets(str, 1, f);
	fscanf(f, "%f,%f,%f,%f,%f", &arr_var[0], &arr_var[1],&arr_var[2],&arr_var[3],&arr_var[4]);
	//printf("%d\n%d\n%d\n%d\n%d\n", arr_var[0], arr_var[1],arr_var[2],arr_var[3],arr_var[4]);
	fclose(f);

	return arr_var;
}

Point* getPointsFromTXT(Point* all_points,int N, int n)
{
    const char* frd = "C:/Users/yakir.giladi/Documents/Visual Studio 2010/Projects/KmeansPPP/Sales_Transactions_Dataset_Weekly.txt";

	FILE* f = fopen(frd, "r+");
	char str[10000];
	float buffer;

	fgets(str, 20, f);
	for (int i=0 ; i<N ; i++) {

		all_points[i].coordinates = (float*)calloc(n,sizeof(float));
		for (int j=0 ; j<n ; j++) {

			fscanf(f, "%f,", &buffer);
			all_points[i].coordinates[j] = buffer;
	
		}
	}
	fclose(f);
	return all_points;
}

void write_results_to_file(Cluster* all_clusters, int num_clusters, int num_coordinates, double qm)
{
	const char* fwt = "C:/Users/yakir.giladi/Documents/Visual Studio 2010/Projects/KmeansPPP/results.txt";
	FILE* f = fopen(fwt, "w"); // open file to rhite inside of it 
	if (f == NULL)
	{
		printf("File did not opened");
	}
	fseek(f, 0, SEEK_SET); // set file point to the start 
	// write data by format
	fprintf(f, "\t\t\t\t\tNumber of best measure clusters\n");
	fprintf(f, "\t\t\t\t\tK = %d	QM = %lf\n", num_clusters, qm);
	fprintf(f, "\t\t\t\t\tCenters of the clusters:\n\n" );

	for (int i = 0 ; i < num_clusters ; i++)
	{
		fprintf(f, "C%d	",i+1);
		for (int j = 0 ; j < num_coordinates ; j++)
		{
			fprintf(f, "%.2f	", all_clusters[i].center.coordinates[j]);
			if ((j+1) % 13 == 0 && (j+1) != 52)
				fprintf(f, "\n	");
			if((j+1) == 52)
				fprintf(f,"\n\n");
		}
	}
	
	fclose(f); //closeing the file
}



double calc_distance_between_2_coordinate(double coor_1, double coor_2, int square) 
{

	//printf("point_1:%f point_2:%f\n", point_1, point_1);

	// ((point_1 - point_2)^2)^0.5
	if (square == 1)
		return sqrt(pow(fabs(coor_1 - coor_2), 2));
	else 
		return pow(fabs(coor_1 - coor_2), 2);
}

double calc_distance_between_2_clusters(Cluster cluster_1, Cluster cluster_2, int num_coordinates)
{
	float* coordinates_arr = (float*)calloc(num_coordinates, sizeof(float)); // init dist_arr real distance = Zero	
	double distance = 0;

	if (parallel) // OMP
	{
		//calcDistanceCoordiantesWithCuda(cluster_1.center.coordinates, cluster_2.center.coordinates, coordinates_arr, num_coordinates); // CUDA
		#pragma omp parallel for
		for (int i = 0 ; i < num_coordinates ; i++)
		{
			distance += coordinates_arr[i];
		}

		//distance = sum_coordinates_arr(coordinates_arr, num_coordinates); // OMP
	}
	else // Sequence
	{
		for (int i = 0 ; i < num_coordinates ; i++)
		{
			distance += coordinates_arr[i];
			//dist_coor_arr += calc_distance_between_2_coordinate(cluster_1.center.coordinates[i], cluster_2.center.coordinates[i], 0);
		}
	}
	free(coordinates_arr);

	//return sqrt(distance); // sequenced
	return distance;
}

double get_distance_between_2_points(Point point_1,Point point_2, int num_coordinates)
{
	double distance = 0;

	if (parallel) // OMP
	{
		float* coordinates_arr = (float*)calloc(num_coordinates, sizeof(float));
		//calcDistanceCoordiantesWithCuda(point_1.coordinates, point_2.coordinates,coordinates_arr, num_coordinates); // CUDA
		
		#pragma omp parallel for
		for (int i = 0 ; i < num_coordinates ; i++)
		{
			coordinates_arr[i] = calc_distance_between_2_coordinate(point_1.coordinates[i], point_2.coordinates[i], 0);
		}
		distance = sum_coordinates_arr(coordinates_arr, num_coordinates); // OMP
		free(coordinates_arr);
		return distance;
	}
	else // Sequence 
	{

		for (int i = 0 ; i < num_coordinates ; i++)
		{
			distance += calc_distance_between_2_coordinate(point_1.coordinates[i], point_2.coordinates[i], 0);
		}
		return sqrt(distance); // Sequence
	}
	
}


void associate_points_to_clusters(Point* all_points, Cluster* all_clusters, int num_points, int num_clusters, int num_coordinates)
{
	float* dist_arr = (float*)calloc(num_clusters, sizeof(float)); // init dist_arr real distance

	//printf("calc_distance_and_cluster_relate\n");
	for (int i = 0; i < num_points; i++) {

		int cluster_related = 0; // cluster_related is the index of the cluster to relate the point
		
		if (parallel) {

			//#pragma omp parallel for // is slower
			for (int j = 0; j < num_clusters; j++) {

				dist_arr[j] = get_distance_between_2_points(all_points[i],all_clusters[j].center, num_coordinates);
				//printf("dist_single_coordinate_final:%f\n",dist_arr[j]); 
			}
			
			cluster_related = calc_min(dist_arr, num_clusters); // Calc min distance
		}
		else // Sequence
		{ 
			//printf("num_point: %d\n",i);
			for (int j = 0; j < num_clusters; j++) {

				dist_arr[j] = get_distance_between_2_points(all_points[i],all_clusters[j].center, num_coordinates);
				//printf("dist_single_coordinate_final:%f\n",dist_arr[j]); 
			}

			cluster_related = calc_min(dist_arr, num_clusters); // Calc min distance
		}
		
		reset_arr(dist_arr, num_clusters);
		
		// Relate the point to the cluster and increase the number of the points in the cluster
		//printf("cluster_related:%d\n",cluster_related);
		//printf("Relate the point to the cluster and increase the number of the points in the cluster\n");

		all_clusters[cluster_related].points_in_cluster[all_clusters[cluster_related].num_points_in_cluster] = all_points[i];
		all_clusters[cluster_related].num_points_in_cluster++;
	}
	free(dist_arr);
}

int calc_min(float* dist_arr, int num_clusters)
{
	int cluster_related = 0;
	double min_distance = dist_arr[0];

	for (int i = 1 ; i < num_clusters ; i++) 
	{
		if (dist_arr[i] < min_distance)
		{ 
			min_distance = dist_arr[i];
			cluster_related = i;
		}
	}

	return cluster_related;
}

void recenter_single_cluster(Cluster* cluster, const int num_coordinates)
{
	Point new_center; // create new center
	new_center.coordinates = (float*)calloc(num_coordinates, sizeof(float));

	for(int i = 0 ; i < cluster->num_points_in_cluster ; i++)
		for (int j = 0 ; j < num_coordinates ; j++)
			new_center.coordinates[j] += cluster->points_in_cluster[i].coordinates[j]; // sum of all the same coordinate

	for(int i = 0 ; i < num_coordinates ; i++)
		new_center.coordinates[i] = new_center.coordinates[i] / cluster->num_points_in_cluster; // doing the avarage

	cluster->center = new_center;
}

void recenter_all_clusters(Cluster* all_clusters,int num_clusters, int num_coordinates)
{
	for (int i = 0 ; i < num_clusters ; i++) // clusters
	{
		recenter_single_cluster(&all_clusters[i],num_coordinates);
	}
}

double calc_diameter_of_single_cluster(Cluster* cluster, int num_coordinates)
{
	double distance = 0, max_distance = 0;

	for (int i = 0 ; i < cluster->num_points_in_cluster ; i++) // clusters
	{
		for (int j = 0 ; j < cluster->num_points_in_cluster ; j++) // clusters
		{
			if( i != j )
			{
				if (parallel) // OMP
				{
					distance = get_distance_between_2_points(cluster->points_in_cluster[i], cluster->points_in_cluster[j], num_coordinates);
					//// CUDA
					//float* distance_arr = (float*)calloc(num_coordinates, sizeof(float));
					//calcDistanceCoordiantesWithCuda(cluster.points_in_cluster[i].coordinates, 
					//	cluster.points_in_cluster[j].coordinates,distance_arr,num_coordinates); 
					//// OMP
					//distance = sum_coordinates_arr(distance_arr,num_coordinates);
					//free(distance_arr);
				}
				else // Sequence
				{
				distance = get_distance_between_2_points(cluster->points_in_cluster[i], cluster->points_in_cluster[j], num_coordinates);//print_logs(distance);
				//printf("distance:%.3f\n",distance);

				}

				if (distance > max_distance) {
					max_distance = distance;
					//printf("max_distance:%.3f\n",max_distance);
				}

				//LOGS
				//print_logs(max_distance);
				
			}
		}

		
	}

	return max_distance;
}

void calc_diameter_of_all_clusters(Cluster* all_clusters,int num_clusters, int num_coordinates)
{
	if (parallel)
	{
		//printf("Calculate diameters with CUDA..\n");
		for (int i = 0; i < num_clusters; i++) 
		{
			all_clusters[i].diameter = calc_diameter_of_single_cluster(&all_clusters[i],num_coordinates);
			//printf("OMP:\n");
			//printf("all_clusters[%d].diameter:%f\n",i,all_clusters[i].diameter);
		}
	}
	else
	{
		// Sequence
		printf("Calculate diameters Iterative..\n");
		for (int i = 0 ; i < num_clusters ; i++) // clusters
		{
			//printf("cluster:%d\n",i);
			all_clusters[i].diameter = calc_diameter_of_single_cluster(&all_clusters[i],num_coordinates);
			//printf("Sequence:\n");
			//printf("all_clusters[%d].diameter:%f\n",i,all_clusters[i].diameter);
		}
	}
}

double sum_coordinates_arr(float* corrdinates, int num_coordinates) 
{
	//omp_set_num_threads(4);
	double sum = 0;

	#pragma omp parallel for reduction(+:sum) // OMP
	for (int i = 0 ; i < num_coordinates ; i++)
	   sum += corrdinates[i];

	return sqrt(sum);
}

float* get_diameter_of_all_clusters (Cluster* all_clusters, int num_clusters)
{
	float* all_diameters = (float*)calloc(num_clusters, sizeof(float));
	
	if (parallel)
	{
		#pragma omp parallel for
		for (int i = 0; i < num_clusters; i++)
			all_diameters[i] = all_clusters[i].diameter; // OMP
	}
	else
	{
		for (int i = 0; i < num_clusters; i++)
			all_diameters[i] = all_clusters[i].diameter;
	}
	return all_diameters;
}

int* get_num_of_points_of_all_clusters (Cluster* all_clusters, int num_clusters)
{
	int* all_num_of_points = (int*)calloc(num_clusters, sizeof(int));

	#pragma omp parallel for
	for (int i = 0; i < num_clusters; i++)
		all_num_of_points[i] = all_clusters[i].num_points_in_cluster; // OMP

	return all_num_of_points;
}

double calc_qm (Cluster* all_clusters,int num_clusters, int num_coordinates)
{
	calc_diameter_of_all_clusters(all_clusters, num_clusters , num_coordinates);
	//int arr_size = num_clusters * (num_clusters - 1);
	//float* qm_arr = (float*)calloc(arr_size, sizeof(float));
	printf("Calculate QM: \n");
	int iterator = 0;
	double semi_q = 0, new_qm = 0;

	for (int i = 0 ; i < num_clusters ; i++)
	{
		double diameter = all_clusters[i].diameter;
		//printf("%d:diameter:%f\n",i,diameter);
		//printf("all_clusters[%d].diameter:%f\n",i,all_clusters[i].diameter);
		for (int j = 0 ; j < num_clusters ; j++)
		{	
			if (i != j) {

				if(parallel)
				{
					// calc distance between centers by Cuda
					float* distance_centers_coordinates = (float*)calloc(num_coordinates, sizeof(float));
					calcDistanceCoordiantesWithCuda(all_clusters[i].center.coordinates, // CUDA
						all_clusters[j].center.coordinates, distance_centers_coordinates, num_coordinates);
					double distance_centers = sum_coordinates_arr(distance_centers_coordinates,num_coordinates); // OMP
					free(distance_centers_coordinates);
					semi_q += diameter / distance_centers;
				}
				else
				{
					//reg
					double distance_centers = get_distance_between_2_points(all_clusters[i].center, all_clusters[j].center, num_coordinates);
					//printf("distance_centers: %f\n",distance_centers);
					semi_q += diameter / distance_centers;
				}
				
				
				iterator++;
			}
			//q = (d1/D12 + d1/D13 + d2/D21 + d2/D23 + d3/D31 + d3/D32) / 6;
		}
	}

	//sum_qm_arr_by_CUDA(qm_arr, semi_q, arr_size); 
	//new_qm = qm_arr[0] / iterator;

	new_qm = semi_q / iterator;

	//printf("SEMI_QM: %f\n",semi_q);
	//printf("ITERATOR: %d\n",iterator);
	//printf("NEW-QM: %f\n",new_qm);

	return semi_q / iterator;
}

void print_logs(int var)
{
	//LOG
	for (int i = 0 ; i < 110; i++) 
		(exp(sin(exp(sin(exp(sin(exp(-2.))))))));
}

bool check_termination_condition(Cluster* all_clusters, int num_clusters, int num_coordinates)
{
	for (int i = 0; i < num_clusters; i++) { // Clusters
		
		for (int j = 0; j < all_clusters[i].num_points_in_cluster; j++) { // Points in Cluster
			
			for (int k = 0; k < num_clusters; k++) { // Other Clusters
			
				if (i != k)
				{

					float point_2_other_center = get_distance_between_2_points(all_clusters[i].points_in_cluster[j], all_clusters[k].center, num_coordinates);
					float point_2_center = get_distance_between_2_points(all_clusters[i].points_in_cluster[j], all_clusters[i].center, num_coordinates);
					
					//printf("point_2_other_center:%f\n",point_2_other_center);
					//printf("point_2_center:%f\n",point_2_center);

					if (point_2_other_center < point_2_center)
					{
						printf("need to recenter\n");
						return false;
					}
				}
			}
		}
	}
	return true;
}

float* reset_arr(float* arr, int size)
{	
	if (parallel) // OMP
	{
		#pragma omp parallel for
		for (int i = 0; i < size; i++)
		  arr[i] = 0;		
	}
	else // Sequenced
	{
		for (int i = 0; i < size; i++)
		  arr[i] = 0;	
	}

	return arr;
}