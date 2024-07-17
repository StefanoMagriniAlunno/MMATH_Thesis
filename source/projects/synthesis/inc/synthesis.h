/**
 * @file synthesis.h
 * @author Stefano Magrini Alunno (stefanomagrini99@gmail.com)
 * @brief
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
 * @brief void function
 *
 * @param[in] in_dset_path: path of input directory
 * @param[in] out_dset_path: path of output directory
 * @param[in] file_path: path of file with all relative path
 * @param[in] log_path: path of log file
 *
 * @return status code:
 * - 0 success
 * - 1 error reading file
 * - 2 error memory allocation
 * - 3 error writing file
 */
int csynthesis(
    const char* const in_dset_path,
    const char* const out_dset_path,
    const char* const file_path,
    const char* const log_path
);
