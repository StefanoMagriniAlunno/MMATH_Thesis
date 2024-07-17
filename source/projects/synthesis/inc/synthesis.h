/**
 * @file synthesis.h
 * @author Stefano Magrini Alunno (stefanomagrini99@gmail.com)
 * @brief definition of status code macros
 * @version 0.0.0
 * @date 2024-07-16
 *
 * @copyright Copyright (c) 2024
 *
 * @todo Implement csynthesis function
 */
#include <stdint.h>
#include <stdlib.h>

/**
 * @brief status code macros
 *
 */
#define SUCCESS 0
#define LOG_FOPEN_MISSED 1
#define LOG_FWRITE_MISSED 2
#define FOPEN_MISSED -1
#define FWRITE_MISSED -2
#define FREAD_MISSED -3
#define MEMORY_ERROR -4
#define SUPER_ERROR 999

/**
 * @brief This function takes a dataset directory and synthetises all images
 *
 * @param[in] in_dset_path: path of input directory
 * @param[in] out_dset_path: path of output directory
 * @param[in] file_path: path of file with all relative path
 * @param[in] log_path: path of log file
 * @param[in] n_threads: number of threads
 *
 * @return int: status code
 *
 * @note only pgm format is supported, this function reports events and errors in a log file.
 */
int csynthesis(
    const char* const in_dset_path,
    const char* const out_dset_path,
    const char* const file_path,
    const char* const log_path,
    int n_threads);
