#include <cuda.h>
#include <fstream>
#include "fcm.h"

std::vector<float> cudafcm(
    const std::vector<float> &data,
    const std::vector<float> &centroids,
    const int n_tails,
    const char *const log_path)
{
    // read gpu properties
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    const char *const device_name = prop.name;  // name of the device
    const int n_sm = prop.multiProcessorCount;  // number of SMs
    const int max_threads_per_block = prop.maxThreadsPerBlock;  // max threads per block
    const int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;  // max threads per SM
    const int max_threadX = prop.maxThreadsDim[0];  // max threads per block in x dimension
    const int max_threadY = prop.maxThreadsDim[1];  // max threads per block in y dimension
    const int max_threadZ = prop.maxThreadsDim[2];  // max threads per block in z dimension
    const int max_blockX = prop.maxGridSize[0];  // max blocks per grid in x dimension
    const int max_blockY = prop.maxGridSize[1];  // max blocks per grid in y dimension
    const int max_blockZ = prop.maxGridSize[2];  // max blocks per grid in z dimension
    const int max_shared_memory_per_block = prop.sharedMemPerBlock;  // max shared memory per block
    const int max_shared_memory_per_sm = prop.sharedMemPerMultiprocessor;  // max shared memory per SM
    const int max_registers_per_block = prop.regsPerBlock;  // max registers per block
    const int max_registers_per_sm = prop.regsPerMultiprocessor;  // max registers per SM
    const int warp_size = prop.warpSize;  // warp size
    const int max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;  // max blocks per SM

    // write log
    std::ofstream log_file(log_path);
    log_file << "Device: " << device_name << std::endl;
    log_file << "Number of SMs: " << n_sm << std::endl;
    log_file << "Max threads per block: " << max_threads_per_block << std::endl;
    log_file << "Max threads per SM: " << max_threads_per_sm << std::endl;
    log_file << "Max threads per block in x dimension: " << max_threadX << std::endl;
    log_file << "Max threads per block in y dimension: " << max_threadY << std::endl;
    log_file << "Max threads per block in z dimension: " << max_threadZ << std::endl;
    log_file << "Max blocks per grid in x dimension: " << max_blockX << std::endl;
    log_file << "Max blocks per grid in y dimension: " << max_blockY << std::endl;
    log_file << "Max blocks per grid in z dimension: " << max_blockZ << std::endl;
    log_file << "Max shared memory per block: " << max_shared_memory_per_block << std::endl;
    log_file << "Max shared memory per SM: " << max_shared_memory_per_sm << std::endl;
    log_file << "Max registers per block: " << max_registers_per_block << std::endl;
    log_file << "Max registers per SM: " << max_registers_per_sm << std::endl;
    log_file << "Warp size: " << warp_size << std::endl;
    log_file << "Max blocks per SM: " << max_blocks_per_sm << std::endl;
    log_file.close();

    std::vector<float> ret(centroids);
    ret[0] = n_tails;
    ret[1] = data[1];

    return ret;
}
