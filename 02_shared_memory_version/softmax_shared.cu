#include <cuda_runtime.h>
#include <iostream>

#define N 10000000

const int threadsPerBlock = 256;
const int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;

// kernel for generating data on the gpu and finding max values per block
__global__ void generate_data(float* x, float* max_partial){
  __shared__ float cache[threadsPerBlock]; // create a shared memory for a block
  int cacheIdx = threadIdx.x; // index for the shared memory
  int idx = threadIdx.x + blockIdx.x * blockDim.x; // global index
  
  float local_max = -INFINITY; // setting local max
  while(idx < N){
    x[idx] = idx * 0.02f + 0.01f; // generating data
    local_max = fmaxf(local_max, x[idx]); // finding local_max
    idx += blockDim.x * gridDim.x; // incrementing index
  } 

  cache[cacheIdx] = local_max; // storing the corresponding local_max into shared memory.

  __syncthreads(); // we wait till all threads reach here and are in sync.

  int stride = blockDim.x / 2; // applying reduction method.
  while(stride != 0){
    if(cacheIdx < stride){ // we compute max for half of the memory.
      cache[cacheIdx] = fmaxf(cache[cacheIdx], cache[cacheIdx + stride]);
    }
    __syncthreads();// wait till all threads reach here and are in sync.
    stride /= 2; // we keep halving the array.
  }

  if(cacheIdx == 0)
    max_partial[blockIdx.x] = cache[0]; // storing each blocks result in max_partial
}

// FIRST-PASS: responsible for calculating sum and exp 
__global__ void first_pass(float* x, float* sum_partial, float max){
  __shared__ float cache[threadsPerBlock]; // create shared memory
  int cacheIdx = threadIdx.x; // index for shared memory
  int idx = threadIdx.x + blockIdx.x * blockDim.x; // global index
  
  float temp = 0.0f; // temp value accumulator
  float val = 0.0f; // temp variable to store exp, so it's computed once and used wherever needed.
  while(idx < N){
    val = expf(x[idx] - max); // we calculate exp, and subtracting max to prevent underflow/overflow
    x[idx] = val; // doing exponentiation in-place
    temp += val; // incrementing value
    idx += blockDim.x * gridDim.x; // incrementing index
  }
  
  cache[cacheIdx] = temp; // storing the accumulated partial sum in the shared memory

  __syncthreads(); // waiting for all threads to sync up

  int stride = blockDim.x / 2; // reduction step
  while(stride != 0){
    if(cacheIdx < stride){
      cache[cacheIdx] = cache[cacheIdx] + cache[cacheIdx + stride];
    }
    __syncthreads();
    stride /= 2;
  }

  if(cacheIdx == 0)
    sum_partial[blockIdx.x] = cache[0]; // store result of each block 
}

// SECOND-PASS: responsible for normalizing teh exp values
__global__ void second_pass(float* x, float* z, float sum){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  while(idx < N){
    z[idx] = x[idx]/sum;
    idx += blockDim.x * gridDim.x;
  }
}

int main(void){
  int size = N * sizeof(float);
  
  // HOST data
  float* h_max_partial = new float[blocksPerGrid]; // for storing partial max
  float* h_sum_partial = new float[blocksPerGrid]; // for storing partial sums
  float* h_z = new float[size]; // for final value
  float max = -INFINITY; 
  float sum = 0.0f;

  // DEVICE data
  float* d_x;
  float* d_z;
  float* d_max_partial;
  float* d_sum_partial;
  
  // allocating memory on device
  cudaMalloc((void**)&d_x, size);
  cudaMalloc((void**)&d_max_partial, blocksPerGrid * sizeof(float));
  cudaMalloc((void**)&d_sum_partial, blocksPerGrid * sizeof(float));
  cudaMalloc((void**)&d_z, size);
  
  //running this kernel, generates data on gpu and also finds max
  generate_data<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_max_partial);
  // finding the final maximum on host side
  cudaMemcpy(h_max_partial, d_max_partial, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < blocksPerGrid; i++){
    max = fmaxf(max, h_max_partial[i]);
  }
  
  //doing the first pass, updates x in place with exp(x) and finds sum of exponents
  first_pass<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_sum_partial, max);
  // finding the final sum
  cudaMemcpy(h_sum_partial, d_sum_partial, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < blocksPerGrid; i++){
    sum += h_sum_partial[i];
  }
  
  // second pass finally normalizes
  second_pass<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_z, sum);
  cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost); // copying the results to HOST from DEVICE
  
  // ensuring for correct results, as result of softmax should sum to 1
  float test = 0.0f;
  for(int i = 0; i < N; i++){
    test += h_z[i];
  }

  printf("SUM-TEST-RESULT: %.3f\n",test);

  // cleaning up resources
  delete[] h_z;
  delete[] h_sum_partial;
  delete[] h_max_partial;

  cudaFree(d_z);
  cudaFree(d_x);
  cudaFree(d_sum_partial);
  cudaFree(d_max_partial);

  return 0;
}
