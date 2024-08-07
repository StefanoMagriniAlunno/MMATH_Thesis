/**
 * @file synthesis.h
 * @author Stefano Magrini Alunno (stefanomagrini99@gmail.com)
 * @brief declaration of csynthesis and status code macros
 *
 * @date 2024-07-16
 *
 * @copyright Copyright (c) 2024
 */
#ifndef SYNTHESIS_H
#define SYNTHESIS_H

#include <stdint.h>
#include <stdlib.h>

/**
 * @brief status code macros
 *
 * @note SUPER_ERROR is an error present but not reported in log
 *
 */
#define SUCCESS 0
#define LOG_ERROR 1
#define IO_ERROR 2
#define VALUE_ERROR 3
#define MEMORY_ERROR 4

/**
 * @brief This function takes a dataset directory and synthetises all images
 *
 * @param in_dset_path: path of input directory
 * @param out_dset_path: path of output directory
 * @param file_path: path of file with all relative path
 * @param n_tiles: size of tiles
 * @param n_threads: number of threads
 * @param log_path: path of log file
 *
 * @return int: status code
 *
 * @note only pgm format is supported, this function reports events and errors
 * in a log file.
 */
int csynthesis (const char *const in_dset_path,
                const char *const out_dset_path, const char *const file_path,
                unsigned n_tiles, unsigned n_threads,
                const char *const log_path);

#endif // SYNTHESIS_H
