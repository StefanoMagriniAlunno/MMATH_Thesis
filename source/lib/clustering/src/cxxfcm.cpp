#include <fstream>
#include <string>
#include <vector>
#include "fcm.h"


int cxxfcm(
    const char *const datafile_path,
    const char *const outfile_path,
    const char *const centroids_path,
    int n_tiles,
    const char *const log_path)
{
    // reset log_file
    {
        std::ofstream log_file(log_path, std::ios::trunc);
        if (!log_file.is_open()) {
            return LOGERROR;
        }
    }

    std::vector<float> data, initial_centroids;

    // read datafile_path
    {
        std::ifstream file(datafile_path, std::ios::binary);
        if (!file.is_open()) {
            // open log_path
            std::ofstream log_file(log_path, std::ios::app);
            if (!log_file.is_open()) {
                return LOGERROR;
            }
            log_file << __FILE__ << " : " << __LINE__ << " details: " << datafile_path << std::endl;
        }

        // Get file size
        file.seekg(0, std::ios::end);
        long unsigned size = file.tellg();
        file.seekg(0, std::ios::beg);
        // Compute number of floats
        long unsigned n_floats = size / sizeof(float);

        // Read data
        data.resize(n_floats);
        file.read(reinterpret_cast<char*>(data.data()), size);
        file.close();
    }

    // read centroids_path
    {
        std::ifstream file(centroids_path, std::ios::binary);
        if (!file.is_open()) {
            // open log_path
            std::ofstream log_file(log_path, std::ios::app);
            if (!log_file.is_open()) {
                return LOGERROR;
            }
            log_file << __FILE__ << " : " << __LINE__ << " details: " << datafile_path << std::endl;
        }

        // Get file size
        file.seekg(0, std::ios::end);
        long unsigned size = file.tellg();
        file.seekg(0, std::ios::beg);
        // Compute number of floats
        long unsigned n_floats = size / sizeof(float);

        // Read data
        initial_centroids.resize(n_floats);
        file.read(reinterpret_cast<char*>(initial_centroids.data()), size);
        file.close();
    }

    // call CUDA function
    try
    {
        std::vector<float> centroids = cudafcm(data, initial_centroids, n_tiles, log_path);

        // save centroids in outfile_path
        {
            std::ofstream file(outfile_path, std::ios::binary);
            if (!file.is_open()) {
                // open log_path
                std::ofstream log_file(log_path, std::ios::app);
                if (!log_file.is_open()) {
                    return LOGERROR;
                }
                log_file << __FILE__ << " : " << __LINE__ << " details: " << outfile_path << std::endl;
            }
            file.write(reinterpret_cast<char*>(centroids.data()), centroids.size() * sizeof(float));
            file.close();
        }

    }
    catch (const std::bad_alloc& e)
    {
        // open log_path
        std::ofstream log_file(log_path, std::ios::app);
        if (!log_file.is_open()) {
            return LOGERROR;
        }
        log_file << __FILE__ << " : " << __LINE__ << " details: " << e.what() << std::endl;
    }

    return SUCCESS;
}
