/**
 * @file cudafcm.cu
 * @author Stefano Magrini Alunno (stefanomagrini99@gmail.com)
 * @brief definition of cudafcm function
 *
 * @date 2024-07-20
 *
 * @copyright Copyright (c) 2024
 */
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <thread>

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform_reduce.h>

#include <cublas_v2.h>

#include <omp.h>

#include "fcm.h"

// reserved memory for the system in GiB (a float castable value)
#define GiB_SYS_RESERVED_MEM 1.0
// max number of threads per block
#define MAX_THREADS_PER_BLOCK 128

// used to pass a log instruction
#define LOGGER(log_stream, level, message)                                    \
  do                                                                          \
    {                                                                         \
      log_stream << __FILE__ << ":" << __LINE__ << " \t" << (level) << " "    \
                 << (message) << std::endl;                                   \
      log_stream.flush ();                                                    \
    }                                                                         \
  while (0)
// check for errors in the runtime
#define CHECK_ERROR_RUNTIME_ERROR(assertion, closing_instruction, message)    \
  if (!(assertion))                                                           \
    {                                                                         \
      LOGGER (log_stream, "ERROR caught", message);                           \
      {                                                                       \
        closing_instruction;                                                  \
      }                                                                       \
      throw std::runtime_error (message);                                     \
    }
// check for errors for bad allocation
#define CHECK_ERROR_BAD_ALLOC(assertion, closing_instruction, message)        \
  if (!(assertion))                                                           \
    {                                                                         \
      LOGGER (log_stream, "ERROR caught", message);                           \
      log_stream.flush ();                                                    \
      {                                                                       \
        closing_instruction;                                                  \
      }                                                                       \
      throw std::bad_alloc ();                                                \
    }
// used to pass an empty instruction
#define PASS_INSTRUCTION ;

/// @cond
float
tocputool (const float *const d_addr)
{
  float value;
  cudaMemcpy (&value, d_addr, sizeof (float), cudaMemcpyDeviceToHost);
  return value;
}
/// @endcond

/**
 * @brief struct used to manage the memory partitions in the device
 * In this way the memory is allocated only once and the data is
 * stored in a contiguous way.
 *
 */
struct partition
{
  float *d_new_centroids; /*!< device memory for new centroids */
  float *d_centroids;     /*!< device memory for centroids */
  float *d_data;       /*!< device memory for data points in a single batch */
  float *d_weights;    /*!< device memory for weights of data points */
  float *d_matrix;     /*!< device memory for matrix of distances */
  float *d_energies;   /*!< device memory for energies of data points */
  size_t batch_size;   /*!< number of data points to use in a single
                          batch */
  size_t n_centroids;  /*!< number of centroids */
  size_t n_dimensions; /*!< number of dimensions */
};

/**
 * @brief used in update_centroids to access data in a non-contiguous
 * way This struct is used to access data in a non-contiguous way
 *
 */
struct non_contiguous_access
{
  float *data;   /*!< address of the data */
  size_t stride; /*!< stride of the data, n_dimensions */

  /*!< constructor */
  __host__ __device__
  non_contiguous_access (float *data, size_t stride)
      : data (data), stride (stride)
  {
  }

  /*!< operator */
  __host__ __device__ float
  operator() (size_t i) const
  {
    return data[i * stride];
  }
};

/**
 * @brief This kernel computes the matrix U2 of membership between
 * data points and centroids
 *
 * @param[in] d_data : the i-th is d_data[i * n_dimensions + k]
 * for k = 0, ..., n_dimensions - 1
 * @param[in] d_weights : the weight of the i-th data point is
 * d_weights[i]
 * @param[in] d_centroids : the j-th is
 * d_centroids[j * n_dimensions + k] for k = 0, ..., n_dimensions - 1
 * @param[out] d_matrix : the weighted membership between the i-th data point
 * and the j-th centroid is stored in d_matrix[i * n_centroids + j]
 * @param[out] d_energies : the energy of the i-th data point is stored in
 * d_energies[i]
 * @param n_data : number of data points
 * @param n_dimensions : dimensions of data points
 * @param n_centroids : number of centroids
 *
 * @details This kernel requires a grid of blocks with n_data blocks
 * and MAX_THREADS_PER_BLOCK threads for each block.
 *
 * @note This kernel synchronize threads at the end of the computation
 */
__global__ void
kernel_compute_U2 (const float *const d_data, const float *const d_weights,
                   const float *const d_centroids, float *const d_matrix,
                   float *const d_energies, size_t n_data, size_t n_dimensions,
                   size_t n_centroids)
{
  __shared__ float sdata[MAX_THREADS_PER_BLOCK];
  size_t i = blockIdx.x;  // i-th data
  size_t j = threadIdx.x; // j-th centroid
  float value = 0;
  float min_value = 0;
  float d2 = 0;

  // compute the distance between the i-th data point and the j-th
  // centroid
  if (i < n_data && j < n_centroids)
    {
      for (size_t k = 0; k < n_dimensions; k++)
        {
          float diff = d_data[i * n_dimensions + k]
                       - d_centroids[j * n_dimensions + k];
          value += diff * diff;
        }
    }
  d2 = value;
  // syncronyze threads of this block
  __syncthreads ();

  // compute the min value of the block
  if (j < n_centroids)
    sdata[j] = value;
  else
    sdata[j] = FLT_MAX;
  __syncthreads ();
  for (size_t s = MAX_THREADS_PER_BLOCK / 2; s > 0; s >>= 1)
    {
      if (j < s && sdata[j] > sdata[j + s])
        sdata[j] = sdata[j + s];
      __syncthreads ();
    }
  min_value = sdata[0];
  // syncronyze threads of this block
  __syncthreads ();

  // prepare the row to a stable normalization
  if (min_value == 0.0)
    {
      // let to 1 the components that are 0 and to 0 the others
      if (i < n_data && j < n_centroids)
        value = value == 0.0 ? 1.0 : 0.0;
    }
  else
    {
      // for each component of the row, assign min/value
      if (i < n_data && j < n_centroids)
        value = min_value / value;
    }
  // syncronyze threads of this block
  __syncthreads ();

  // compute the sum of the row
  if (j < n_centroids)
    sdata[j] = value;
  else
    sdata[j] = 0.0;
  __syncthreads ();
  for (size_t s = MAX_THREADS_PER_BLOCK / 2; s > 0; s >>= 1)
    {
      if (j < s)
        sdata[j] += sdata[j + s];
      __syncthreads ();
    }
  min_value = sdata[0];
  // syncronyze threads of this block
  __syncthreads ();

  // assign the value to the matrix
  if (i < n_data && j < n_centroids) {
    value /= min_value;
    d_matrix[i * n_centroids + j] = value * value * d_weights[i];
  }
  // syncronyze threads of this block
  __syncthreads ();

  // compute energy
  if (i < n_data && j < n_centroids)
    value = d_matrix[i * n_centroids + j] * d2;  // compute partial energy
  // syncronyze threads of this block
  __syncthreads ();

  // compute the sum of the partial energies
  if (j < n_centroids)
    sdata[j] = value;
  else
    sdata[j] = 0.0;
  __syncthreads ();
  for (size_t s = MAX_THREADS_PER_BLOCK / 2; s > 0; s >>= 1)
    {
      if (j < s)
        sdata[j] += sdata[j + s];
      __syncthreads ();
    }
  value = sdata[0];  // energy of the data point
  // syncronyze threads of this block
  __syncthreads ();

  // assign the energy to the matrix
  if (i < n_data && j == 0)
    d_energies[i] = value;
  // syncronyze threads of this block
  __syncthreads ();
}

/**
 * @brief Update centroids yet allocated on the device
 *
 * @param[in] d_data : data points stored as d_data[i * n_dimensions +
 * k] for k = 0, ..., n_dimensions - 1
 * @param[in] d_matrix : matrix of distances stored as d_matrix[i *
 * n_centroids
 * + j]
 * @param[out] h_centroids_weight : weights of centroids
 * @param[out] d_new_centroids : new centroids
 * @param n_data : number of data points
 * @param n_dimensions : dimensions of data points
 * @param n_centroids : number of centroids
 * @param prop : properties of the device
 * @param handle : cublas handle
 * @param log_stream : log file
 *
 * @exception std::runtime_error : if an error occurs during the
 * computation
 *
 * @note This function synchronize threads at the end of the
 * computation
 */
__host__ void
update_centroids (const float *const d_data, float *const d_matrix,
                  float *const h_centroids_weight,
                  float *const d_new_centroids, size_t n_data,
                  size_t n_dimensions, size_t n_centroids,
                  const cudaDeviceProp &prop, cublasHandle_t handle,
                  std::ofstream &log_stream)
{
  // use cublas to compute the new centroids
  cublasStatus_t status;
  cudaError_t err;

  // compute the product of d_matrix and d_data
  float alpha = 1.0;
  float beta = 1.0;
  // for  k=1:n_dimensions, j=1:n_centroids
  // d_new_centroids[k + j*n_dimensions] = d_new_centroids[k +
  // j*n_dimensions]
  // + sum_i=1:n_data d_data[k + i*n_dimensions]*d_matrix[j +
  // i*n_centroids]
  status = cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_T, n_dimensions,
                        n_centroids, n_data, &alpha, d_data, n_dimensions,
                        d_matrix, n_centroids, &beta, d_new_centroids,
                        n_dimensions);

  // check error
  CHECK_ERROR_RUNTIME_ERROR (status == CUBLAS_STATUS_SUCCESS, PASS_INSTRUCTION,
                             "CUBLAS sgemm failed with status "
                                 + std::to_string (status));

  /// @remark it's not necessary to syncronize threads

  for (size_t i = 0; i < n_centroids; i++)
    {
      // compute the sum of d_matrix along data points
      try
        {
          float *d_matrix_ptr = thrust::raw_pointer_cast (d_matrix + i);
          auto begin = thrust::make_transform_iterator (
              thrust::counting_iterator<size_t> (0),
              non_contiguous_access (d_matrix_ptr, n_centroids));
          auto end = begin + n_data;
          h_centroids_weight[i] += thrust::transform_reduce (
              begin, end, thrust::identity<float> (), 0.0f,
              thrust::plus<float> ());
        }
      catch (std::runtime_error &e)
        {
          LOGGER (log_stream, "CRITICAL caught", e.what ());
          throw std::runtime_error (e.what ());
        }
    }

  // syncronize threads
  err = cudaDeviceSynchronize ();
  // check for errors
  CHECK_ERROR_RUNTIME_ERROR (err == cudaSuccess, PASS_INSTRUCTION,
                             cudaGetErrorString (err));
}

/**
 * @brief This function compute the matrix of distances between data
 * points and centroids
 *
 * @param[in] d_data : data points stored as d_data[i * n_dimensions +
 * k] for k = 0, ..., n_dimensions - 1
 * @param[in] d_weights : weights of data points
 * @param[in] d_centroids : centroids stored as d_centroids[j *
 * n_dimensions + k] for k = 0, ..., n_dimensions - 1
 * @param[out] d_matrix : matrix of distances stored as d_matrix[i *
 * n_centroids + j] for j = 0, ..., n_centroids - 1
 * @param n_data : number of data points
 * @param n_dimensions : dimensions of data points
 * @param n_centroids : number of centroids
 * @param prop : properties of the device
 * @param log_stream : log file
 *
 * @exception std::runtime_error : if an error occurs during the
 * computation
 *
 * @note This function synchronize threads at the end of the
 * computation
 */
__host__ void
compute_U2 (const float *const d_data, const float *const d_weights,
            const float *const d_centroids, float *const d_matrix,
            float *const d_energies, size_t n_data, size_t n_dimensions,
            size_t n_centroids, const cudaDeviceProp &prop, std::ofstream &log_stream)
{
  cudaError_t err;

  // each block works on a single data point
  dim3 grid (n_data);
  dim3 block (MAX_THREADS_PER_BLOCK);

  // call the kernel
  // clang-format off
  kernel_compute_U2<<<grid, block>>> (d_data, d_weights, d_centroids, d_matrix, d_energies, n_data,
                                        n_dimensions, n_centroids);
  // clang-format on

  // check for errors
  err = cudaGetLastError ();
  CHECK_ERROR_RUNTIME_ERROR (err == cudaSuccess, PASS_INSTRUCTION,
                             cudaGetErrorString (err));

  // synchronize threads
  err = cudaDeviceSynchronize ();
  // check for errors
  CHECK_ERROR_RUNTIME_ERROR (err == cudaSuccess, PASS_INSTRUCTION,
                             cudaGetErrorString (err));
}

/**
 * @brief This function compute centroids
 *
 * @param data : data points stored as data[i * n_dimensions + k] for
 * k = 0,
 * ..., n_dimensions - 1
 * @param weights : weights of data points
 * @param partitions : partitions of the data points
 * @param prop : properties of the device
 * @param log_stream : log file
 * @return float : energy of the data points
 *
 * @details For a better performance, the log messages are written
 * only in case of error.
 *
 * @exception std::runtime_error : if an error occurs during the
 * computation
 * @exception std::bad_alloc : if an error occurs during the memory
 * allocation
 */
__host__ float
compute_centroids (const std::vector<float> &data,
                   const std::vector<float> &weights,
                   const struct partition partitions,
                   const cudaDeviceProp &prop, std::ofstream &log_stream)
{
  // cicle over the batches
  cudaError_t err;
  cublasStatus_t status;
  cublasHandle_t handle;

  // initialize cublas for U2 computation
  status = cublasCreate (&handle);
  // check for errors
  CHECK_ERROR_RUNTIME_ERROR (status == CUBLAS_STATUS_SUCCESS, PASS_INSTRUCTION,
                             "CUBLAS initialization failed with status "
                                 + std::to_string (status));

  // d_new_centroids is a zero-initialized vector
  err = cudaMemset (partitions.d_new_centroids, 0,
                    partitions.n_centroids * partitions.n_dimensions
                        * sizeof (float));
  // check for errors
  CHECK_ERROR_RUNTIME_ERROR (err == cudaSuccess, cublasDestroy (handle),
                             cudaGetErrorString (err));

  // allocate memory for the centroids weight
  float *h_centroids_weight
      = (float *)calloc (partitions.n_centroids, sizeof (float));
  // check for errors
  CHECK_ERROR_BAD_ALLOC (h_centroids_weight != NULL, cublasDestroy (handle),
                         "Memory allocation failed");

  // define energy of the data points
  float energy = 0;

  // cicle over the data points
  size_t n_data = data.size () / partitions.n_dimensions;
  size_t c_data = 0; // completed data points
  while (c_data < n_data)
    {
      size_t batch_size = std::min (partitions.batch_size, n_data - c_data);

      // report completion
      LOGGER (log_stream, "INFO",
              "completed " + std::to_string (c_data) + " ("
                  + std::to_string (100 * (double)c_data / (double)n_data)
                  + "%) data points");

      // copy data to the device
      err = cudaMemcpy (partitions.d_data,
                        data.data () + c_data * partitions.n_dimensions,
                        batch_size * partitions.n_dimensions * sizeof (float),
                        cudaMemcpyHostToDevice);
      // check for errors
      CHECK_ERROR_RUNTIME_ERROR (
          err == cudaSuccess,
          {
            cublasDestroy (handle);
            free (h_centroids_weight);
          },
          cudaGetErrorString (err));

      // copy weights to the device
      err = cudaMemcpy (partitions.d_weights, weights.data () + c_data,
                        batch_size * sizeof (float), cudaMemcpyHostToDevice);
      // check for errors
      CHECK_ERROR_RUNTIME_ERROR (
          err == cudaSuccess,
          {
            cublasDestroy (handle);
            free (h_centroids_weight);
          },
          cudaGetErrorString (err));

      try
        {
          // compute the matrix U2
          compute_U2 (partitions.d_data, partitions.d_weights,
                      partitions.d_centroids, partitions.d_matrix, partitions.d_energies,
                      batch_size, partitions.n_dimensions, partitions.n_centroids, prop,
                      log_stream);

          // update the new centroids
          update_centroids (partitions.d_data, partitions.d_matrix,
                            h_centroids_weight, partitions.d_new_centroids,
                            batch_size, partitions.n_dimensions,
                            partitions.n_centroids, prop, handle, log_stream);
          // compute batch energy as the sum of the energies
          // use thrust::transform_reduce
          float *batch_energy_ptr = thrust::raw_pointer_cast (partitions.d_energies);
          auto begin = thrust::make_transform_iterator (
              thrust::counting_iterator<size_t> (0),
              non_contiguous_access (batch_energy_ptr, 1));
          auto end = begin + batch_size;
          float batch_energy = thrust::transform_reduce (
              begin, end, thrust::identity<float> (), 0.0f, thrust::plus<float> ());
          // update the total energy
          energy += batch_energy;
        }
      catch (std::runtime_error &e)
        {
          LOGGER (log_stream, "CRITICAL caught", e.what ());
          {
            cublasDestroy (handle);
            free (h_centroids_weight);
          }
          throw std::runtime_error (e.what ());
        }

      c_data += batch_size;
    }

  // divide d_new_centroids by the centroids weight
  for (size_t i = 0; i < partitions.n_centroids; i++)
    {
      if (h_centroids_weight[i] != 0)
        {
          // use cublas to divide d_new_centroids[i,:] by
          // h_centroids_weight[i]
          float alpha = 1.0 / h_centroids_weight[i];
          status = cublasSscal (
              handle, partitions.n_dimensions, &alpha,
              partitions.d_new_centroids + i * partitions.n_dimensions, 1);
          // check for errors
          CHECK_ERROR_RUNTIME_ERROR (
              status == CUBLAS_STATUS_SUCCESS,
              {
                cublasDestroy (handle);
                free (h_centroids_weight);
              },
              "CUBLAS scal failed with status " + std::to_string (status));
        }
    }

  // syncronize threads
  err = cudaDeviceSynchronize ();
  // check for errors
  CHECK_ERROR_RUNTIME_ERROR (
      err == cudaSuccess,
      {
        cublasDestroy (handle);
        free (h_centroids_weight);
      },
      cudaGetErrorString (err));

  // free data
  status = cublasDestroy (handle);
  free (h_centroids_weight);

  return energy;
}

__host__ std::vector<float>
cudafcm (const std::vector<float> &data, const std::vector<float> &weights,
         const std::vector<float> &centroids, size_t n_dimensions,
         float tollerance, std::ofstream &log_stream)
{
  // check gpu properties
  int device_count;
  int device;
  cudaError_t err;
  cudaDeviceProp prop;

  // Inizializza CUDA
  err = cudaGetDeviceCount (&device_count);
  // check for errors
  CHECK_ERROR_RUNTIME_ERROR (err == cudaSuccess, PASS_INSTRUCTION,
                             cudaGetErrorString (err));
  // check if there are devices
  CHECK_ERROR_RUNTIME_ERROR (device_count > 0, PASS_INSTRUCTION,
                             "No CUDA devices found");
  err = cudaSetDevice (0);
  // check for errors
  CHECK_ERROR_RUNTIME_ERROR (err == cudaSuccess, PASS_INSTRUCTION,
                             cudaGetErrorString (err));
  err = cudaGetDevice (&device);
  // check for errors
  CHECK_ERROR_RUNTIME_ERROR (err == cudaSuccess, PASS_INSTRUCTION,
                             cudaGetErrorString (err));

  // get gpu properties
  err = cudaGetDeviceProperties (&prop, device);

  // check for errors
  CHECK_ERROR_RUNTIME_ERROR (err == cudaSuccess, PASS_INSTRUCTION,
                             cudaGetErrorString (err));
  // check if num of centroids is less then threads per block
  CHECK_ERROR_RUNTIME_ERROR (
      centroids.size () / n_dimensions <= prop.maxThreadsPerBlock,
      PASS_INSTRUCTION,
      "Number of centroids is greater than threads per block");
  // check if max num of threads per block is correct
  CHECK_ERROR_RUNTIME_ERROR (
      prop.maxThreadsPerBlock >= MAX_THREADS_PER_BLOCK, PASS_INSTRUCTION,
      "Number of threads per block is less than required");
  // check if shared memory is enough
  CHECK_ERROR_RUNTIME_ERROR (
      prop.sharedMemPerBlock >= MAX_THREADS_PER_BLOCK * sizeof (float),
      PASS_INSTRUCTION, "Shared memory is less than required");

  // report on the log file
  LOGGER (log_stream, "INFO", "Using device: " + std::string (prop.name));
  LOGGER (log_stream, "DEBUG",
          "Size of data points: " + std::to_string (n_dimensions));
  LOGGER (log_stream, "DEBUG", "Tollerance: " + std::to_string (tollerance));
  LOGGER (log_stream, "DEBUG",
          "Number of centroids: "
              + std::to_string (centroids.size () / n_dimensions));
  LOGGER (log_stream, "DEBUG",
          "Number of data points: "
              + std::to_string (data.size () / n_dimensions));

  // check if number of centroids is supported
  CHECK_ERROR_RUNTIME_ERROR(
      centroids.size () / n_dimensions <= MAX_THREADS_PER_BLOCK,
      PASS_INSTRUCTION, "Number of centroids not supported");

  // prepare cuda context:
  size_t batch_size = 0; // number of data points to use in a single batch
  float *d_main_ptr;     // pointer to the main memory pool
  {
    // reserved memory for system
    size_t reserved_memory
        = (float)(GiB_SYS_RESERVED_MEM) * 1024 * 1024 * 1024;
    // set a memory pool for the device over the remaining memory
    size_t free, total;
    cudaMemGetInfo (&free, &total);
    size_t pool_memory = free - reserved_memory;
    LOGGER (log_stream, "INFO",
            "Total memory: " + std::to_string (total) + " bytes");
    LOGGER (log_stream, "INFO",
            "Free memory: " + std::to_string (free) + " bytes");
    /**
     * @remark
     * C is number of centroids
     * D is number of dimensions
     * N is number of data points
     * The required memory is:
     *   2 x C x D + N x D + 2 x N + N x C
     * = 2 x C x D + (D + 2 + C) x N
     * The number of used data points is:
     *   (pool_memory - 2 x C x D) / (D + 2 + C)
     */

    // compute the number of data points to use
    batch_size = (pool_memory - 2 * centroids.size () * sizeof (float))
                 / ((n_dimensions + centroids.size () / n_dimensions + 2)
                    * sizeof (float));
    batch_size = std::min (batch_size, data.size () / n_dimensions);

    // check if batch_size is less than 0
    CHECK_ERROR_BAD_ALLOC (batch_size > 0, PASS_INSTRUCTION,
                           "Not enough memory to allocate the data points");
    // check if batch_size is greater than max number of activable
    // blocks
    CHECK_ERROR_BAD_ALLOC (
        batch_size <= prop.maxGridSize[0], PASS_INSTRUCTION,
        "Number of data points is greater than max number of "
        "activable blocks");

    LOGGER (log_stream, "INFO", "batch_size: " + std::to_string (batch_size));

    // compute the total memory required
    size_t total_memory
        = (2 * centroids.size ()
           + batch_size
                 * (n_dimensions + 2 + centroids.size () / n_dimensions))
          * sizeof (float);

    // try to allocate memory
    err = cudaMalloc (&d_main_ptr, total_memory);
    // check for errors
    CHECK_ERROR_BAD_ALLOC (err == cudaSuccess, PASS_INSTRUCTION,
                           cudaGetErrorString (err));
  }

  // prepare data
  struct partition partitions = {
    .d_new_centroids = d_main_ptr,                 // len = centroids.size()
    .d_centroids = d_main_ptr + centroids.size (), // len = centroids.size()
    .d_data
    = d_main_ptr + 2 * centroids.size (), // len = batch_size * n_dimensions
    .d_weights = d_main_ptr + 2 * centroids.size ()
                 + batch_size * n_dimensions, // len = batch_size
    .d_matrix = d_main_ptr + 2 * centroids.size () + batch_size * n_dimensions
                + batch_size, // len = batch_size * centroids.size() /
                              // n_dimensions
    .d_energies = d_main_ptr + 2 * centroids.size () + batch_size * n_dimensions
                  + batch_size + batch_size * centroids.size ()
                      / n_dimensions, // len = batch_size
    .batch_size = batch_size,
    .n_centroids = centroids.size () / n_dimensions,
    .n_dimensions = n_dimensions,
  };

  // copy centroids to the device
  err = cudaMemcpy (partitions.d_centroids, centroids.data (),
                    centroids.size () * sizeof (float),
                    cudaMemcpyHostToDevice);
  // check for errors
  CHECK_ERROR_RUNTIME_ERROR (err == cudaSuccess, cudaFree (d_main_ptr),
                             cudaGetErrorString (err));

  // update centroids
  try
    {
      float pred_energy;                    // energy of the previous iteration
      float energy = INFINITY;              // current energy
      do
        {
          pred_energy = energy;  // safe energy in pred_energy

          // update centroids
          energy = compute_centroids (data, weights, partitions, prop,
                                            log_stream);
          LOGGER (log_stream, "INFO",
                  "energy: " + std::to_string (energy));

          // move d_new_centroids to d_centroids
          err = cudaMemcpy (partitions.d_centroids, partitions.d_new_centroids,
                            centroids.size () * sizeof (float),
                            cudaMemcpyDeviceToDevice);
          // check for errors
          CHECK_ERROR_RUNTIME_ERROR (err == cudaSuccess, cudaFree (d_main_ptr),
                                     cudaGetErrorString (err));

          // check stability
          if (pred_energy < energy) {
            LOGGER(log_stream, "WARNING", "Numerical stability error");
          }
        }
      while (energy >= tollerance && pred_energy - energy >= tollerance);
    }
  catch (std::bad_alloc &e)
    {
      LOGGER (log_stream, "CRITICAL caught", e.what ());
      cudaFree (d_main_ptr);
      throw std::runtime_error (e.what ());
    }
  catch (std::runtime_error &e)
    {
      LOGGER (log_stream, "CRITICAL caught", e.what ());
      cudaFree (d_main_ptr);
      throw std::runtime_error (e.what ());
    }

  std::vector<float> out_centroids;

  // copy centroids from the device
  out_centroids.resize (centroids.size ());
  err = cudaMemcpy (out_centroids.data (), partitions.d_centroids,
                    centroids.size () * sizeof (float),
                    cudaMemcpyDeviceToHost);
  // check for errors
  CHECK_ERROR_RUNTIME_ERROR (err == cudaSuccess, cudaFree (d_main_ptr),
                             cudaGetErrorString (err));

  // free memory
  cudaFree (d_main_ptr);

  return out_centroids;
}
