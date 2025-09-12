#include <cuda_runtime.h>
#include <iostream>

#define N 100000000

// method for generating data on GPU.
__global__ void generate_data(float* d_data){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N){
    d_data[idx] = (idx * 0.0002f + 0.0001f) - ((N-1) * 0.0002f + 0.0001f);// subtracting with max
  }
}

// FIRST_PASS: calculates the exponent of each value in the vector.
__global__ void first_pass(float* d_data, float* d_exps){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N){
    d_exps[idx] = expf(d_data[idx]);
  }
}

// SECOND_PASS: calculates the sum of exponents.
__global__ void second_pass(float* d_exps, float* d_sum){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N){
    atomicAdd(d_sum, d_exps[idx]);
  }
}

// THIRD_PASS: normalizes the exponents by dividing them by sum calculated in previous pass, thus calculating the softmax.
__global__ void third_pass(float* d_exps, float* d_smax, float* d_sum){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N){
    d_smax[idx] = d_exps[idx]/(*d_sum);
  }
}

// method to pring device properties.
int get_device_properties(){
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop,0);
  printf("---------- DEVICE-PROPERTIES ----------\n");
  printf("Name: %s\n",prop.name);
  printf("Version: %d.%d\n",prop.major,prop.minor);
  printf("ClockRate: %d\n",prop.clockRate);
  printf("Total Global Memory: %ld\n",prop.totalGlobalMem);
  printf("Shared Memory Per Block: %ld\n",prop.sharedMemPerBlock);
  printf("Registers Per Block: %d\n",prop.regsPerBlock);
  printf("Max Threads Per Block: %d\n",prop.maxThreadsPerBlock);
  printf("----------------------------------------\n\n");

  if(prop.maxThreadsPerBlock * sizeof(float) >= prop.sharedMemPerBlock)
    return 256;
  else
    return prop.maxThreadsPerBlock;
}

// method to perform validity of softmax
void check_validity(float* smax){
  float sum = 0.0f;

  for(unsigned int i = 0; i < N; i++){
    sum += smax[i];
  }
  
  std::cout<<"----- Performing validity check -----"<<std::endl;
  std::cout<<"ERROR: "<<(1.0f-sum)<<"%"<<std::endl;
}

int main(void){
  float milliseconds = 0.0f;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // get properties of the cuda-device
  // parameters for number of blocks and threads for calling a kernel
  int blocksSize = get_device_properties();
  int numBlocks = (N + blocksSize - 1)/blocksSize;

  // size for allocating memory in bytes
  int size = N * sizeof(float);

  float sum = 0.0f; // sum on host
  float* h_smax = new float[N]; // softmax array on host

  float* d_sum; // sum on device
  float* d_data; // data array on device
  float* d_exps; // exp array on device
  float* d_smax; // softmax array on device
  
  // allocating memory 
  cudaMalloc((void**)&d_sum, sizeof(float));
  cudaMalloc((void**)&d_data, size);
  cudaMalloc((void**)&d_exps, size);
  cudaMalloc((void**)&d_smax, size);
  
  
  cudaEventRecord(start);
  
  // copying sum from host to device sum
  cudaMemcpy(d_sum, &sum, sizeof(float), cudaMemcpyHostToDevice);

  // populate data
  generate_data<<<numBlocks, blocksSize>>>(d_data);

  //first-pass
  first_pass<<<numBlocks, blocksSize>>>(d_data, d_exps);
  //second-pass
  second_pass<<<numBlocks, blocksSize>>>(d_exps, d_sum);
  //third-pass
  third_pass<<<numBlocks, blocksSize>>>(d_exps, d_smax, d_sum);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_smax, d_smax, size, cudaMemcpyDeviceToHost);
  
  check_validity(h_smax);
  printf("\nELAPSED-TIME: %f ms",milliseconds);
  
  cudaFree(d_sum);
  cudaFree(d_smax);
  cudaFree(d_exps);
  cudaFree(d_data);
  
  delete[] h_smax;

  return 0;
}
