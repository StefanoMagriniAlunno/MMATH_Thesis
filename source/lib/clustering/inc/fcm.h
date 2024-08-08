/**
 * @file fcm.h
 * @author Stefano Magrini Alunno (stefanomagrini99@gmail.com)
 * @brief declaration of cxxfcm and status code macros
 *
 * @date 2024-07-16
 *
 * @copyright Copyright (c) 2024
 */
#pragma once

#include <fstream>
#include <vector>

#define SUCCESS 0
#define LOG_ERROR 1
#define IO_ERROR 2
#define DEVICE_ERROR 3

/**
 * @brief This function takes a datafile path and compute an fcm clustering
 *
 * @param datafile_path: path of datafile
 * @param weightsfile_path: path of weights file
 * @param centroids_path: path of file with initial centroids
 * @param outfile_centroids_path: path of output file
 * @param n_dimensions: dimension of data points
 * @param tollerance: tollerance of the algorithm
 * @param log_path: path of log file
 *
 * @return int: status code
 */
int cxxfcm (const char *const datafile_path,
            const char *const weightsfile_path,
            const char *const centroids_path,
            const char *const outfile_centroids_path, size_t n_dimensions,
            float tollerance, const char *const log_path);

/**
 * @brief This function takes a datafile path and compute an fcm clustering
 *
 * @param data: vector of data
 * @param weights: vector of weights
 * @param centroids: vector of initial centroids
 * @param n_dimensions: dimension of data points
 * @param tollerance: tollerance of the algorithm
 * @param log_stream: log stream
 *
 * @return std::vector<float>: vector of centroids
 *
 * @exception std::bad_alloc : if the size of data is not valid
 *
 * @details This function maximize the usage of the GPU memory and threads
 * to compute the fcm clustering.
 * @section FCM algorithm
 * Fuzzy C-Means (FCM) is a method of clustering which allows one piece of data
 * to belong to two or more clusters. Each data point in the dataset is
 * assigned a membership value, which is a number between 0 and 1. The sum of
 * the membership values for each data point is 1. U[i][j] is the membership
 * value of the i-th data point for the j-th cluster. U[i][j] is proportional
 * to 1/dist(i,j)^2 where dist(i,j) is the distance between the i-th data point
 * and the j-th cluster. so U[i][j] = 1/dist(i,j)^2 / sum(1/dist(i,j)^2) for
 * all j. Then the centroids are updated as the weighted average of the data
 * points. C[j] = sum(U[i][j]^2 * X[i]) / sum(U[i][j]^2) for all i. The
 * algorithm stops when the centroids do not change significantly (under a
 * certain tollerance).
 */
std::vector<float>
cudafcm (const std::vector<float> &data, const std::vector<float> &weights,
         const std::vector<float> &centroids, size_t n_dimensions,
         float tollerance, std::ofstream &log_stream);
