/**
 * @file csynthesis.c
 * @author Stefano Magrini Alunno (stefanomagrini99@gmail.com)
 * @brief definition of synthesis
 * @version 0.0.0
 * @date 2024-07-16
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <linux/limits.h>
#include "synthesis.h"

#define MESSAGE_MAXLEN 256

/**
 * @brief This support function performs a single synthesis.
 *
 * @param[in] in_path : complete path of the image
 * @param[in] out_path : complete path of the output
 * @param[out] log_message : message for log when there is an error
 * @return int: status code
 *
 */
int _csynthesis(
    const char* const in_path,
    const char* const out_path,
    char log_message[MESSAGE_MAXLEN])
{
    long int width = 0;
    long int height = 0;
    long int size = 0;
    double * data = NULL;
    unsigned char * byte_board = NULL;
    FILE* in_file = fopen(in_path, "r");
    if (in_file == NULL)
    {
        sprintf(log_message, "%s : %d details: %s", __FILE__, __LINE__, in_path);
        return FOPEN_MISSED;
    }
    if (fscanf(in_file, "P5\n%ld %ld\n255", &width, &height) != 2)
    {
        fclose(in_file);
        sprintf(log_message, "%s : %d", __FILE__, __LINE__);
        return FREAD_MISSED;
    }
    fgetc(in_file);  // used to skip the character newline
    size = width * height;
    byte_board = (unsigned char *)calloc((size_t)size, sizeof(unsigned char));
    if (byte_board == NULL)
    {
        fclose(in_file);
        sprintf(log_message, "%s : %d details: %ld", __FILE__, __LINE__, (long)((size_t)size*sizeof(unsigned char)));
        return MEMORY_ERROR;
    }
    if ((int64_t)fread(byte_board, sizeof(unsigned char), (size_t)size, in_file) != size)
    {
        fclose(in_file);
        free(byte_board);
        sprintf(log_message, "%s : %d details: %ld", __FILE__, __LINE__, (long)(sizeof(unsigned char)*(size_t)size));
        return FREAD_MISSED;
    }
    fclose(in_file);

    data = (double *)calloc((size_t)size, sizeof(double));
    if (data == NULL)
    {
        free(byte_board);
        sprintf(log_message, "%s : %d details: %ld", __FILE__, __LINE__, (long)((size_t)size*sizeof(double)));
        return MEMORY_ERROR;
    }

    ///@todo operate the synthesis

    FILE* out_file = fopen(out_path, "w");
    if (out_file == NULL)
    {
        free(byte_board);
        free(data);
        sprintf(log_message, "%s : %d details: %s", __FILE__, __LINE__, out_path);
        return FOPEN_MISSED;
    }
    fprintf(out_file, "P5\n%ld %ld\n255\n", width, height);
    if ((int64_t)fwrite(byte_board, sizeof(unsigned char), (size_t)size, out_file) != size)
    {
        fclose(out_file);
        free(byte_board);
        free(data);
        sprintf(log_message, "%s : %d details: %ld", __FILE__, __LINE__, (long)(sizeof(unsigned char)*(size_t)size));
        return FWRITE_MISSED;
    }
    fclose(out_file);
    free(byte_board);
    free(data);

    return SUCCESS;
}


int csynthesis(
    const char* const in_db_path,
    const char* const out_db_path,
    const char* const list_file_path,
    const char* const log_file_path,
    int n_threads)
{
    /**
     * @note in_db_path and out_db_path exist
     * @note list_file_path is a file with paths of files relative to in_db_path
     * @note it's possible to create files in out_db_path with the same relative path
     * @note log_file_path is a file where to write the log, with following format
     * "%d %d %s", thread_num, status, file_path
     */

    FILE* log_file = fopen(log_file_path, "w");
    if (log_file == NULL)
    {
        return LOG_FOPEN_MISSED;
    }
    FILE* list_file = fopen(list_file_path, "r");
    if (list_file == NULL)
    {
        fclose(log_file);
        return FOPEN_MISSED;
    }

    /**
     * @brief Generate a group of threads to:
     * - read the file list_file_path
     * - make the synthesis
     * - report the result in log_file_path
     */
    int error_detected = SUCCESS;
    #pragma omp parallel num_threads(n_threads) shared(error_detected)
    {
        char path[PATH_MAX];
        while (fgets(path, sizeof(path), list_file) != NULL && error_detected == SUCCESS)
        {
            path[strlen(path)-1] = '\0';  // l'ultimo carattere era "\n"
            char log_message[MESSAGE_MAXLEN];
            char in_path[PATH_MAX];
            char out_path[PATH_MAX];
            int state = 1;
            strcpy(in_path, in_db_path);
            strcat(in_path, "/");
            strcat(in_path, path);
            strcpy(out_path, out_db_path);
            strcat(out_path, "/");
            strcat(out_path, path);
            int ret = _csynthesis(in_path, out_path, log_message);
            #pragma omp critical
            {
                if (ret != SUCCESS)
                {
                    int chars_written = fprintf(log_file, "%d in %s\n", ret, log_message);
                    if (chars_written < 0)
                    {
                        error_detected = SUPER_ERROR;
                    }
                    error_detected = ret;
                }
                else
                {
                    int chars_written = fprintf(log_file, "%d %d %s\n", omp_get_thread_num(), state, path);
                    if (chars_written < 0)
                    {
                        error_detected = LOG_FWRITE_MISSED;
                    }
                }
            }
        }
    }
    if (error_detected != SUCCESS)
    {
        fclose(log_file);
        fclose(list_file);
        return error_detected;
    }

    return SUCCESS;
}
