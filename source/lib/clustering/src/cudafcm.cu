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
#define GiB_SYS_RESERVED_MEM 0.5

float
tocputool (const float *const d_addr)
{
  float value;
  cudaMemcpy (&value, d_addr, sizeof (float), cudaMemcpyDeviceToHost);
  return value;
}

/**
 * @brief struct used to manage the memory partitions in the device
 *
 */
struct partition
{
  float *d_centroids;     /*!< device memory for centroids */
  float *d_new_centroids; /*!< device memory for new centroids */
  float *d_data;       /*!< device memory for data points in a single batch */
  float *d_matrix;     /*!< device memory for matrix of distances */
  size_t batch_size;   /*!< number of data points to use in a single batch */
  size_t n_centroids;  /*!< number of centroids */
  size_t n_dimensions; /*!< number of dimensions */
};

/**
 * @brief used in compute_centroids to access data in a non-contiguous way
 *
 */
struct non_contiguous_access
{
  float *data;
  size_t stride;

  __host__ __device__
  non_contiguous_access (float *data, size_t stride)
      : data (data), stride (stride)
  {
  }

  __host__ __device__ float
  operator() (size_t i) const
  {
    return data[i * stride];
  }
};

struct unary_op
{
  __host__ __device__ void
  operator() (float &x) const
  {
    x = float (x == 0);
  }
};

/**
 * @brief This kernel computes the division of the new centroids by the
 * centroids weight
 *
 * @param d_new_centroids : new centroids
 * @param d_centroids : centroids
 * @param n_centroids : number of centroids
 * @param n_dimensions : number of dimensions
 *
 * @details This kernel computes the division of the new centroids by the
 * centroids weight The thread (j, k) computes the division of the k-th
 * dimension of the j-th centroid The storage of data is:
 * - the j-th centroid is stored in d_centroids[j * n_dimensions + k] for k =
 * 0, ..., n_dimensions - 1
 */
__global__ void
kernel_compute_divide (float *const d_new_centroids,
                       const float *const d_centroids, size_t n_centroids,
                       size_t n_dimensions)
{
  size_t j = blockIdx.x;
  size_t k = threadIdx.x;
  if (j < n_centroids && k < n_dimensions)
    d_new_centroids[k + j * n_dimensions] /= d_centroids[j];
}

/**
 * @brief This kernel computes the matrix of distances between data points and
 * centroids
 *
 * @param d_data
 * @param d_centroids
 * @param d_matrix
 * @param n_data
 * @param n_dimensions
 * @param n_centroids
 *
 * @details This kernel computes the matrix of distances between data points
 * and centroids The thread (i, j) computes the distance between the i-th data
 * point and the j-th centroid The storage of data is:
 *  - the i-th data point is stored in d_data[i * n_dimensions + k] for k = 0,
 * ..., n_dimensions - 1
 *  - the j-th centroid is stored in d_centroids[j * n_dimensions + k] for k =
 * 0, ..., n_dimensions - 1
 *  - the distance between the i-th data point and the j-th centroid is stored
 * in d_matrix[i * n_centroids + j] The grid of blocks has in x the number of
 * data points and in y the number of centroids The grid of threads has in x
 * the number of data points and in y the number of centroids
 */
__global__ void
kernel_compute_D2 (const float *const d_data, const float *const d_centroids,
                   float *const d_matrix, size_t n_data, size_t n_dimensions,
                   size_t n_centroids)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n_data && j < n_centroids)
    {
      float distance = 0;
      for (size_t k = 0; k < n_dimensions; k++)
        {
          float diff = d_data[i * n_dimensions + k]
                       - d_centroids[j * n_dimensions + k];
          distance += diff * diff;
        }
      d_matrix[i * n_centroids + j] = distance;
    }
}

/**
 * @brief Update centroids yet allocated on the device
 *
 *
 * @details This function update centroids yet allocated on the device
 * The storage of data is:
 * - the i-th data point is stored in d_data[i * n_dimensions + k] for k = 0,
 * ..., n_dimensions - 1
 * - the j-th centroid is stored in d_centroids[j * n_dimensions + k] for k =
 * 0, ..., n_dimensions - 1
 * - the membership between the i-th data point and the j-th centroid is stored
 * in d_matrix[i * n_centroids + j]
 */
__host__ void
update_centroids (const float *d_data, float *const d_matrix,
                  float *const h_centroids_weight,
                  float *const d_new_centroids, size_t n_data,
                  size_t n_dimensions, size_t n_centroids,
                  const cudaDeviceProp &prop, cublasHandle_t handle,
                  std::ofstream &log_stream)
{
  // use cublas to compute the new centroids
  cublasStatus_t status;

  // compute the product of d_matrix and d_data
  // d_new_centroids += d_matrix^T * d_data
  float alpha = 1.0;
  float beta = 1.0;
  // for k=1:n_dimensions, j=1:n_centroids
  // d_new_centroids[k + j*n_dimensions] = d_new_centroids[k + j*n_dimensions]
  // + sum_i=1:n_data d_data[k+i*n_dimensions]*d_matrix[j+i*n_centroids]
  status = cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_T, n_dimensions,
                        n_centroids, n_data, &alpha, d_data, n_dimensions,
                        d_matrix, n_centroids, &beta, d_new_centroids,
                        n_dimensions);
  if (status != CUBLAS_STATUS_SUCCESS)
    {
      log_stream << "in file " << __FILE__ << " at line " << __LINE__
                 << " : CRITICAL caught " << status << std::endl;
      log_stream.flush ();
      throw std::runtime_error (std::to_string (status));
    }

  for (size_t i = 0; i < n_centroids; i++)
    {
      // compute the sum of d_matrix[:, j]
      float *d_matrix_ptr = thrust::raw_pointer_cast (d_matrix);
      auto begin = thrust::make_transform_iterator (
          thrust::counting_iterator<size_t> (0),
          non_contiguous_access (d_matrix_ptr, n_centroids));
      auto end = begin + n_data;
      h_centroids_weight[i]
          = thrust::transform_reduce (begin, end, thrust::identity<float> (),
                                      0.0f, thrust::plus<float> ());
    }
}

/**
 * @brief Compute the matrix of membership
 *
 * @param d_matrix : matrix of distances
 * @param n_data : number of data points
 * @param n_centroids : number of centroids
 * @param prop : properties of the device
 * @param handle : cublas handle
 * @param log_stream : log file
 *
 * @details This function compute the matrix of membership
 * Save in d_matrix_sum the sum of inverse distances
 * Divide each row of d_matrix by the corresponding value in d_matrix_sum
 * The storage of data is:
 * - the distance between the i-th data point and the j-th centroid is stored
 * in d_matrix[i * n_centroids + j]
 * - the membership between the i-th data point and the j-th centroid is stored
 * in d_matrix[i * n_centroids + j]
 */
__host__ void
compute_U2 (float *const d_matrix, size_t n_data, size_t n_centroids,
            const cudaDeviceProp &prop, cublasHandle_t handle,
            std::ofstream &log_stream)
{
  // use thrust and cublas to compute the sum of each row of d_matrix and
  // normalize it
  cudaError_t err;
  cublasStatus_t status;
  thrust::device_ptr<float> d_matrix_ptr (d_matrix);

#pragma omp parallel for num_threads(prop.multiProcessorCount)                \
    shared(d_matrix_ptr, h_values) schedule(static)
  for (size_t i = 0; i < n_data; i++)
    {
      // compute min value
      float value = thrust::reduce (d_matrix_ptr + i * n_centroids,
                                    d_matrix_ptr + (i + 1) * n_centroids,
                                    FLT_MAX, thrust::minimum<float> ());
      err = cudaGetLastError ();
      if (err != cudaSuccess)
        {
          log_stream << "in file " << __FILE__ << " at line " << __LINE__
                     << " : CRITICAL caught " << cudaGetErrorString (err)
                     << std::endl;
          log_stream.flush ();
          throw std::runtime_error (cudaGetErrorString (err));
        }
      if (value == 0)
        {
          // set to 1 the components that are 0 and at 0 the others
          thrust::for_each (d_matrix_ptr + i * n_centroids,
                            d_matrix_ptr + (i + 1) * n_centroids, unary_op ());
          err = cudaGetLastError ();
          if (err != cudaSuccess)
            {
              log_stream << "in file " << __FILE__ << " at line " << __LINE__
                         << " : CRITICAL caught " << cudaGetErrorString (err)
                         << std::endl;
              log_stream.flush ();
              throw std::runtime_error (cudaGetErrorString (err));
            }
        }
      else
        {
          // set to value/C all C in the row
          thrust::transform (d_matrix_ptr + i * n_centroids,
                             d_matrix_ptr + (i + 1) * n_centroids,
                             thrust::constant_iterator<float> (value),
                             d_matrix_ptr + i * n_centroids,
                             thrust::divides<float> ());

          err = cudaGetLastError ();
          if (err != cudaSuccess)
            {
              log_stream << "in file " << __FILE__ << " at line " << __LINE__
                         << " : CRITICAL caught " << cudaGetErrorString (err)
                         << std::endl;
              log_stream.flush ();
              throw std::runtime_error (cudaGetErrorString (err));
            }
        }
      // normalize all values
      value = thrust::reduce (d_matrix_ptr + i * n_centroids,
                              d_matrix_ptr + (i + 1) * n_centroids);
      err = cudaGetLastError ();
      if (err != cudaSuccess)
        {
          log_stream << "in file " << __FILE__ << " at line " << __LINE__
                     << " : CRITICAL caught " << cudaGetErrorString (err)
                     << std::endl;
          log_stream.flush ();
          throw std::runtime_error (cudaGetErrorString (err));
        }
      value = 1.0 / value;
      status = cublasSscal (handle, n_centroids, &value,
                            d_matrix + i * n_centroids, 1);
      if (status != CUBLAS_STATUS_SUCCESS)
        {
          log_stream << "in file " << __FILE__ << " at line " << __LINE__
                     << " : CRITICAL caught " << status << std::endl;
          log_stream.flush ();
          throw std::runtime_error (std::to_string (status));
        }
    }
}

/**
 * @brief This function compute the matrix of distances between data points and
 * centroids
 *
 * @param d_data : data points
 * @param d_centroids : centroids
 * @param d_matrix : matrix of distances
 * @param n_data : number of data points
 * @param n_dimensions : dimensions of data points
 * @param n_centroids : number of centroids
 * @param prop : properties of the device
 * @param log_stream : log file
 *
 * @details This function compute the matrix of distances between data points
 * and centroids The storage of data is:
 * - the i-th data point is stored in d_data[i * n_dimensions + k] for k = 0,
 * ..., n_dimensions - 1
 * - the j-th centroid is stored in d_centroids[j * n_dimensions + k] for k =
 * 0, ..., n_dimensions - 1
 * - the distance between the i-th data point and the j-th centroid is stored
 * in d_matrix[i * n_centroids + j]
 */
__host__ void
compute_D2 (const float *d_data, const float *d_centroids,
            float *const d_matrix, size_t n_data, size_t n_dimensions,
            size_t n_centroids, const cudaDeviceProp &prop,
            std::ofstream &log_stream)
{
  // compute the coverage of a single block (sqrt(maxThreadsPerBlock),
  // sqrt(maxThreadsPerBlock))
  size_t max_coverage_size = sqrt (prop.maxThreadsPerBlock);

  // compute num of blocks along x and y
  size_t n_blocks_x = (n_data + max_coverage_size - 1) / max_coverage_size;
  size_t n_blocks_y
      = (n_centroids + max_coverage_size - 1) / max_coverage_size;

  // make the grid of blocks
  dim3 grid (n_blocks_x, n_blocks_y);
  dim3 block (max_coverage_size, max_coverage_size);

  // call the kernel
  kernel_compute_D2<<<grid, block> > > (d_data, d_centroids, d_matrix, n_data,
                                        n_dimensions, n_centroids);

  // check for errors
  cudaError_t err = cudaGetLastError ();
  if (err != cudaSuccess)
    {
      log_stream << "in file " << __FILE__ << " at line " << __LINE__
                 << " : CRITICAL caught " << cudaGetErrorString (err)
                 << std::endl;
      log_stream.flush ();
      throw std::runtime_error (cudaGetErrorString (err));
    }

  // synchronize threads
  cudaDeviceSynchronize ();
  err = cudaGetLastError ();
  if (err != cudaSuccess)
    {
      log_stream << "in file " << __FILE__ << " at line " << __LINE__
                 << " : CRITICAL caught " << cudaGetErrorString (err)
                 << std::endl;
      log_stream.flush ();
      throw std::runtime_error (cudaGetErrorString (err));
    }

  return;
}

/**
 * @brief This function compute centroids
 *
 * @param data : data points
 * @param partitions : partitions of the data points
 * @param prop : properties of the device
 * @param log_stream : log file
 * @return float : variation of centroids
 *
 * @details This function update centroids yet allocated on the device
 * Allocate memory for a vector N x n_dimensions & N x n_centroids.
 * For a better performance, the log messages are written only in case of
 * error.
 */
__host__ float
compute_centroids (const std::vector<float> &data,
                   const struct partition partitions,
                   const cudaDeviceProp &prop, std::ofstream &log_stream)
{
  // cicle over the batches
  cudaError_t err;

  // initialize cublas for U2 computation
  cublasHandle_t handle;
  cublasCreate (&handle);
  err = cudaGetLastError ();
  if (err != cudaSuccess)
    {
      log_stream << "in file " << __FILE__ << " at line " << __LINE__
                 << " : CRITICAL caught " << cudaGetErrorString (err)
                 << std::endl;
      log_stream.flush ();
      throw std::runtime_error (cudaGetErrorString (err));
    }

  // allocate memory for the centroids weight
  float *h_centroids_weight
      = (float *)calloc (partitions.n_centroids, sizeof (float));
  size_t n_data = data.size () / partitions.n_dimensions;
  size_t c_data = 0;
  while (c_data < n_data)
    {
      size_t batch_size = std::min (partitions.batch_size, n_data - c_data);
      log_stream << "in file " << __FILE__ << " at line " << __LINE__
                 << " : INFO completed " << c_data << " ("
                 << (int)(100 * (double)c_data / (double)n_data)
                 << "%) data points" << std::endl;
      log_stream.flush ();
      // copy data to the device
      err = cudaMemcpy (partitions.d_data,
                        data.data () + c_data * partitions.n_dimensions,
                        batch_size * partitions.n_dimensions * sizeof (float),
                        cudaMemcpyHostToDevice);
      if (err != cudaSuccess)
        {
          log_stream << "in file " << __FILE__ << " at line " << __LINE__
                     << " : CRITICAL caught " << cudaGetErrorString (err)
                     << std::endl;
          log_stream.flush ();
          throw std::runtime_error (cudaGetErrorString (err));
        }
      // compute the matrix of distances
      compute_D2 (partitions.d_data, partitions.d_centroids,
                  partitions.d_matrix, batch_size, partitions.n_dimensions,
                  partitions.n_centroids, prop, log_stream);
      // compute the matrix of membership
      compute_U2 (partitions.d_matrix, partitions.batch_size,
                  partitions.n_centroids, prop, handle, log_stream);
      // update the new centroids
      update_centroids (partitions.d_data, partitions.d_matrix,
                        h_centroids_weight, partitions.d_new_centroids,
                        batch_size, partitions.n_dimensions,
                        partitions.n_centroids, prop, handle, log_stream);

      c_data += batch_size;
    }
  // write d_new_centroids over d_centroids
  err = cudaMemcpy (partitions.d_centroids, partitions.d_new_centroids,
                    partitions.n_centroids * partitions.n_dimensions
                        * sizeof (float),
                    cudaMemcpyDeviceToDevice);
  if (err != cudaSuccess)
    {
      log_stream << "in file " << __FILE__ << " at line " << __LINE__
                 << " : CRITICAL caught " << cudaGetErrorString (err)
                 << std::endl;
      log_stream.flush ();
      throw std::runtime_error (cudaGetErrorString (err));
    }
  // move h_centroids_weight to the device over new_centroids
  err = cudaMemcpy (partitions.d_new_centroids, h_centroids_weight,
                    partitions.n_centroids * sizeof (float),
                    cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    {
      log_stream << "in file " << __FILE__ << " at line " << __LINE__
                 << " : CRITICAL caught " << cudaGetErrorString (err)
                 << std::endl;
      log_stream.flush ();
      throw std::runtime_error (cudaGetErrorString (err));
    }
  // define a grid of blocks with n_centroids blocks and n_dimensions threads
  // for each block
  dim3 block_grid (partitions.n_centroids, 1);
  dim3 thread_grid (partitions.n_dimensions, 1);
  // compute the new centroids
  kernel_compute_divide<<<block_grid, thread_grid> > > (
      partitions.d_new_centroids, partitions.d_centroids,
      partitions.n_centroids, partitions.n_dimensions);

  // compute the variation of centroids
  float delta_update = 0;

  // free the memory allocated for h_centroids_weight with calloc
  free (h_centroids_weight);

  return delta_update;
}

__host__ std::vector<float>
cudafcm (const std::vector<float> &data, const std::vector<float> &centroids,
         size_t n_dimensions, float tollerance, std::ofstream &log_stream)
{
  // check gpu properties
  int device_count;
  int device;
  cudaError_t err;
  cudaDeviceProp prop;

  // Inizializza CUDA
  err = cudaGetDeviceCount (&device_count);
  if (err != cudaSuccess)
    {
      log_stream << "in file " << __FILE__ << " at line " << __LINE__
                 << " : CRITICAL caught " << cudaGetErrorString (err)
                 << std::endl;
      throw std::runtime_error (cudaGetErrorString (err));
    }
  if (device_count == 0)
    {
      log_stream << "in file " << __FILE__ << " at line " << __LINE__
                 << " : CRITICAL No CUDA devices found" << std::endl;
      log_stream.flush ();
      throw std::runtime_error ("No CUDA devices found");
    }
  err = cudaSetDevice (0);
  if (err != cudaSuccess)
    {
      log_stream << "in file " << __FILE__ << " at line " << __LINE__
                 << " : CRITICAL caught " << cudaGetErrorString (err)
                 << std::endl;
      log_stream.flush ();
      throw std::runtime_error (cudaGetErrorString (err));
    }
  err = cudaGetDevice (&device);
  if (err != cudaSuccess)
    {
      log_stream << "in file " << __FILE__ << " at line " << __LINE__
                 << " : CRITICAL caught " << cudaGetErrorString (err)
                 << std::endl;
      log_stream.flush ();
      throw std::runtime_error (cudaGetErrorString (err));
    }

  // get gpu properties
  err = cudaGetDeviceProperties (&prop, device);
  if (err != cudaSuccess)
    {
      log_stream << "in file " << __FILE__ << " at line " << __LINE__
                 << " : CRITICAL caught " << cudaGetErrorString (err)
                 << std::endl;
      log_stream.flush ();
      throw std::runtime_error (cudaGetErrorString (err));
    }

  // report on the log file
  log_stream << "in file " << __FILE__ << " at line " << __LINE__
             << " : INFO Using device: " << prop.name << std::endl;
  log_stream << "in file " << __FILE__ << " at line " << __LINE__
             << " : DEBUG Size of data points: " << n_dimensions << std::endl;
  log_stream << "in file " << __FILE__ << " at line " << __LINE__
             << " : DEBUG Tollerance: " << tollerance << std::endl;
  log_stream << "in file " << __FILE__ << " at line " << __LINE__
             << " : DEBUG Number of centroids: "
             << centroids.size () / n_dimensions << std::endl;
  log_stream << "in file " << __FILE__ << " at line " << __LINE__
             << " : DEBUG Number of data points: "
             << data.size () / n_dimensions << std::endl;
  log_stream.flush ();

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
    size_t pool_memory = total - reserved_memory;
    log_stream << "in file " << __FILE__ << " at line " << __LINE__
               << " : DEBUG Total memory: " << total << std::endl;
    log_stream << "in file " << __FILE__ << " at line " << __LINE__
               << " : DEBUG Free memory: " << free << std::endl;
    log_stream.flush ();
    /**
     * @brief
     * C is number of centroids
     * D is number of dimensions
     * N is number of data points
     * The required memory is:
     *   2 x C x D + N x D + N x C
     * The number of used data points is:
     *   (pool_memory - 2 x C x D) / (2 x D + C)
     */

    // compute the number of data points to use
    batch_size = (pool_memory - 2 * centroids.size () * sizeof (float))
                 / ((n_dimensions + centroids.size () / n_dimensions)
                    * sizeof (float));
    if (batch_size <= 0)
      {
        log_stream
            << "in file " << __FILE__ << " at line " << __LINE__
            << " : CRITICAL Not enough memory to allocate the data points"
            << std::endl;
        log_stream.flush ();
        throw std::bad_alloc ();
      }
    log_stream << "in file " << __FILE__ << " at line " << __LINE__
               << " : INFO Batch size: " << batch_size << std::endl;
    log_stream.flush ();

    // compute the total memory required
    size_t total_memory = (2 * centroids.size () + batch_size * n_dimensions
                           + batch_size * (centroids.size () / n_dimensions))
                          * sizeof (float);

    // try to allocate memory
    err = cudaMalloc (&d_main_ptr, total_memory);
    if (err != cudaSuccess)
      {
        log_stream << "in file " << __FILE__ << " at line " << __LINE__
                   << " : CRITICAL caught " << cudaGetErrorString (err)
                   << std::endl;
        log_stream.flush ();
        throw std::bad_alloc ();
      }
  }

  // prepare data
  struct partition partitions = {
    .d_centroids = d_main_ptr, // len = centroids.size()
    .d_new_centroids
    = d_main_ptr + centroids.size (), // len = centroids.size()
    .d_data
    = d_main_ptr + 2 * centroids.size (), // len = batch_size * n_dimensions
    .d_matrix
    = d_main_ptr + 2 * centroids.size ()
      + batch_size * n_dimensions, // len = batch_size * centroids.size()
    .batch_size = batch_size,
    .n_centroids = centroids.size () / n_dimensions,
    .n_dimensions = n_dimensions,
  };

  // copy centroids to the device
  err = cudaMemcpy (partitions.d_centroids, centroids.data (),
                    centroids.size () * sizeof (float),
                    cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    {
      log_stream << "in file " << __FILE__ << " at line " << __LINE__
                 << " : CRITICAL caught " << cudaGetErrorString (err)
                 << std::endl;
      log_stream.flush ();
      cudaFree (d_main_ptr);
      throw std::runtime_error (cudaGetErrorString (err));
    }

  // update centroids
  try
    {
      float delta_update = nan (""); // variation of centroids
      do
        {
          // d_new_centroids is a zero-initialized vector
          cudaMemset (partitions.d_new_centroids, 0,
                      centroids.size () * sizeof (float));

          // update centroids
          delta_update
              = compute_centroids (data, partitions, prop, log_stream);
          log_stream << "in file " << __FILE__ << " at line " << __LINE__
                     << " : INFO Variation of centroids: " << delta_update
                     << std::endl;
          log_stream.flush ();

          // move d_new_centroids to d_centroids
          err = cudaMemcpy (partitions.d_centroids, partitions.d_new_centroids,
                            centroids.size () * sizeof (float),
                            cudaMemcpyDeviceToDevice);
          if (err != cudaSuccess)
            {
              log_stream << "in file " << __FILE__ << " at line " << __LINE__
                         << " : CRITICAL caught " << cudaGetErrorString (err)
                         << std::endl;
              log_stream.flush ();
              cudaFree (d_main_ptr);
              throw std::runtime_error (cudaGetErrorString (err));
            }
        }
      while (delta_update > tollerance);
    }
  catch (std::bad_alloc &e)
    {
      log_stream << "in file " << __FILE__ << " at line " << __LINE__
                 << " : CRITICAL caught " << e.what () << std::endl;
      log_stream.flush ();
      cudaFree (d_main_ptr);
      throw std::runtime_error (e.what ());
    }
  catch (std::runtime_error &e)
    {
      log_stream << "in file " << __FILE__ << " at line " << __LINE__
                 << " : CRITICAL caught " << e.what () << std::endl;
      log_stream.flush ();
      cudaFree (d_main_ptr);
      throw std::runtime_error (e.what ());
    }

  // copy centroids from the device
  std::vector<float> new_centroids (centroids.size ());
  err = cudaMemcpy (new_centroids.data (), partitions.d_centroids,
                    centroids.size () * sizeof (float),
                    cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
    {
      log_stream << "in file " << __FILE__ << " at line " << __LINE__
                 << " : CRITICAL caught " << cudaGetErrorString (err)
                 << std::endl;
      log_stream.flush ();
      cudaFree (d_main_ptr);
      throw std::runtime_error (cudaGetErrorString (err));
    }

  // free memory
  cudaFree (d_main_ptr);

  return new_centroids;
}
