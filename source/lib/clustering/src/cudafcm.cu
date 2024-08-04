#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/system_error.h>
#include "fcm.h"


/**
 * @brief This function update centroids yet allocated on the device
 *
 * @param data : data points
 * @param d_centroids : centroids
 * @param d_centroids_support : support for centroids
 * @param n_dimensions : number of dimensions of the data points
 * @param prop : properties of the device
 * @param log_path : log file
 * @return float : variation of centroids
 *
 * @details This function update centroids yet allocated on the device
 * Allocate memory for a vector N x n_dimensions & N x n_centroids.
 */
float update_centroids(
    const std::vector<float> &data,
    const thrust::device_vector<float> &d_centroids,
    const thrust::device_vector<float> &d_centroids_support,
    size_t n_dimensions,
    const cudaDeviceProp &prop,
    std::ofstream &log_path)
{
    // check remaining memory and alloc memory for a vector N x n_dimensions & N x n_centroids
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    log_path << "in file " << __FILE__ << " at line " << __LINE__ << " : DEBUG Free memory: " << free << std::endl;
    log_path << "in file " << __FILE__ << " at line " << __LINE__ << " : DEBUG Total memory: " << total << std::endl;
    size_t data_total_row = data.size() / n_dimensions;
    size_t data_max_partial_row = (free + (n_dimensions + d_centroids.size()) * sizeof(float) - 1) / ((n_dimensions + d_centroids.size()) * sizeof(float));
    // alloc memory for a vector N x n_dimensions & N x n_centroids
    try
    {
        thrust::device_vector<float> d_data_partial;
        d_data_partial.reserve(data_max_partial_row * n_dimensions);
        thrust::device_vector<float> d_centroids_partial;
        d_centroids_partial.reserve(d_centroids.size()*data_max_partial_row);
        for (size_t data_begin_row = 0; data_begin_row < data_total_row; data_begin_row += data_max_partial_row)
        {
            // compute how many rows to load
            size_t data_partial_row = std::min(data_max_partial_row, data_total_row - data_begin_row);
            // resizing vectors
            d_data_partial.resize(data_partial_row * n_dimensions);
            d_centroids_partial.resize(data_partial_row * d_centroids.size());
            //load data_partial_row rows from data
            thrust::copy(data.begin() + data_begin_row * n_dimensions, data.begin() + (data_begin_row + data_partial_row) * n_dimensions, d_data_partial.begin());
            /** @todo compute distances */
            /** @todo compute membership */
            /** @todo compute centroids */
        }
        return 0.0;
    }
    catch (thrust::system_error &e)
    {
        log_path << "in file " << __FILE__ << " at line " << __LINE__ << " : CRITICAL catched " << e.what() << std::endl;
        log_path.flush();
        throw std::runtime_error(e.what());
    }
    catch (std::runtime_error &e)
    {
        log_path << "in file " << __FILE__ << " at line " << __LINE__ << " : CRITICAL catched " << e.what() << std::endl;
        log_path.flush();
        throw std::runtime_error(e.what());
    }
}

std::vector<float> cudafcm(
    const std::vector<float> &data,
    const std::vector<float> &centroids,
    size_t n_dimensions,
    float tollerance,
    std::ofstream &log_stream)
{
    // check gpu properties
    int device_count;
    int device;
    cudaError_t err;
    cudaDeviceProp prop;

    // Inizializza CUDA
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        log_stream << "in file " << __FILE__ << " at line " << __LINE__ << " : CRITICAL caught " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(cudaGetErrorString(err));
    }

    err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        log_stream << "in file " << __FILE__ << " at line " << __LINE__ << " : CRITICAL caught " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(cudaGetErrorString(err));
    }
    if (device_count == 0) {
        log_stream << "in file " << __FILE__ << " at line " << __LINE__ << " : CRITICAL No CUDA devices found" << std::endl;
        throw std::runtime_error("No CUDA devices found");
    }

    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        log_stream << "in file " << __FILE__ << " at line " << __LINE__ << " : CRITICAL caught " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // get gpu properties
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        log_stream << "in file " << __FILE__ << " at line " << __LINE__ << " : CRITICAL caught " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // report on the log file
    log_stream << "in file " << __FILE__ << " at line " << __LINE__ << " : INFO Using device: " << prop.name << std::endl;
    log_stream << "in file " << __FILE__ << " at line " << __LINE__ << " : DEBUG Size of data points: " << n_dimensions << std::endl;
    log_stream << "in file " << __FILE__ << " at line " << __LINE__ << " : DEBUG Tollerance: " << tollerance << std::endl;
    log_stream << "in file " << __FILE__ << " at line " << __LINE__ << " : DEBUG Number of centroids: " << centroids.size() / n_dimensions << std::endl;
    log_stream << "in file " << __FILE__ << " at line " << __LINE__ << " : DEBUG Number of data points: " << data.size() / n_dimensions << std::endl;

    std::vector<float> ret(centroids.size());
    try
    {
        // move centroids to the device using a thrust vector
        thrust::device_vector<float> d_centroids(centroids);
        // reserve memory for an empty vector of centroids
        thrust::device_vector<float> d_centroids_support;
        d_centroids_support.reserve(d_centroids.size());
        // update centroids
        float delta_update = nan("");  // variation of centroids
        do {
            delta_update = update_centroids(data, d_centroids, d_centroids_support, n_dimensions, prop, log_stream);
        } while (delta_update > tollerance);
        // move centroids back to the host
        thrust::copy(d_centroids.begin(), d_centroids.end(), ret.begin());
    }
    catch (thrust::system_error &e)
    {
        log_stream << "in file " << __FILE__ << " at line " << __LINE__ << " : CRITICAL caught " << e.what() << std::endl;
        log_stream.flush();
        throw std::runtime_error(e.what());
    }
    catch (std::runtime_error &e)
    {
        log_stream << "in file " << __FILE__ << " at line " << __LINE__ << " : CRITICAL caught " << e.what() << std::endl;
        log_stream.flush();
        throw std::runtime_error(e.what());
    }

    return ret;
}
