#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <stdio.h>

using namespace std;

void init(int arr[], int n){
//	srand((unsigned)time(0));
	srand(0);
	for(int i=0;i<n;i++)
		arr[i]=rand()%1234 + 1;
}

void swap(int arr[], int pos1, int pos2){
	int temp;
	temp = arr[pos1];
	arr[pos1] = arr[pos2];
	arr[pos2] = temp;
}

int partition(int arr[], int low, int high, int pivot){
	int j = low;

	for(int i=low;i<=high;i++){
		if(arr[i]<=pivot){
			swap(arr, i, j);
			j++;
		}
	}
	return j-1;
}

int pivot(int* array, int first, int last){
	int p = first;
	int pivotElement = array[first];
	for(int i=first+1;i<=last;i++){
		if(array[i]<=pivotElement){
			p++;
			swap(array[i],array[p]);
		}
	}
	swap(array[p],array[first]);
	return p;
}

void quickSort(int arr[], int low, int high){
	if(low < high){
		int pivot = arr[high];
		int pos = partition(arr, low, high, pivot);
		
		quickSort(arr, low, pos-1);
		quickSort(arr, pos+1, high);
	}
}

bool checkFn(int * arr,int n){
	for(int i=0;i<n-1;i++){
		if(arr[i]>arr[i+1]){
			cout << "array[" << i << "] > array[" << i+1 << "]" << endl;
			return false;
		}
	}
	return true;
}

int main(int argc, char *argv[]){
	int n;
	int *arr;
	double t1,t2;
	int debug=1;
	if(argc < 2){
		n=6000000;
	}else{
		n=atoi(argv[1]);
	}
	arr = new int[n];
	init(arr, n);

	t1=omp_get_wtime();
	quickSort(arr, 0 , n-1);
	t2=omp_get_wtime()-t1;

	cout << "quicksort took " << t2 << " sec. to complete" << endl;
	if (!checkFn(arr, n)) {
		cout << "validation failed!" << endl;
	}

	if (debug) {
		cout<<"The sorted array is: ";
		for( int i = 0 ; i < n; i++){
			cout<< arr[i]<<" ";
		}
		cout << endl;
	}

	delete [] arr;
	
}
