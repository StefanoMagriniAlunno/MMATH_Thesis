/**
 * @file csynthesis.c
 * @author Stefano Magrini Alunno (stefanomagrini99@gmail.com)
 * @brief definition of synthesis
 *
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
 * @brief This function performs the synthesis of a single image.
 *
 * @param[in] cells: array of double
 * @param[in] byte_board: array of unsigned char
 * @param[in] width: width of image
 * @param[in] height: height of image
 * @param[in] n_tails: size of tails
 */
void _synthetizer(
    unsigned char * const cells,
    const unsigned char * const byte_board,
    const unsigned width,
    const unsigned height,
    const unsigned n_tails)
{
    unsigned width_short = width - n_tails + 1, height_short = height - n_tails + 1;

    for (unsigned row = 0; row < height; ++row) for (unsigned col = 0; col < width; ++col)  // fisso il pixel che voglio inserire
    {
        long unsigned index_in = row*width + col;
        for (unsigned i = 0; i < n_tails; ++i) for (unsigned j = 0; j < n_tails; ++j)  // identifico dove inserirlo
        {
            if (row >= i && col >= j && row - i < height_short && col - j < width_short)
            {
                long unsigned index_out = (((row-i)*width_short + (col-j))*n_tails + i)*n_tails + j;  // [row-i, col-j, i,j]
                cells[index_out] = byte_board[index_in];
            }
        }
    }
}

/**
 * @brief This support function read an image and call the synthetizer.
 *
 * @param[in] in_path : complete path of the image
 * @param[in] out_path : complete path of the output
 * @param[out] log_message : message for log when there is an error
 * @param[in] n_tails : size of tails
 * @return int: status code
 *
 */
int _reader(
    const char* const in_path,
    const char* const out_path,
    char log_message[MESSAGE_MAXLEN],
    unsigned n_tails)
{
    unsigned width = 0;
    unsigned height = 0;
    long unsigned size = 0;
    unsigned char * cells = NULL;
    unsigned char * byte_board = NULL;
    FILE* in_file = fopen(in_path, "r");
    if (in_file == NULL)
    {
        sprintf(log_message, "%s : %d details: %s", __FILE__, __LINE__, in_path);
        return FOPEN_MISSED;
    }
    if (fscanf(in_file, "P5\n%u %u\n255", &width, &height) != 2)
    {
        fclose(in_file);
        sprintf(log_message, "%s : %d", __FILE__, __LINE__);
        return FREAD_MISSED;
    }
    if (width < n_tails && height < n_tails)
    {
        fclose(in_file);
        sprintf(log_message, "%s : %d details: w=%u, h=%u, n_tails=%u", __FILE__, __LINE__, width, height, n_tails);
        return VALUE_ERROR;
    }
    fgetc(in_file);  // used to skip the character newline
    size = width * height;
    byte_board = (unsigned char *)calloc(size, sizeof(unsigned char));
    if (byte_board == NULL)
    {
        fclose(in_file);
        sprintf(log_message, "%s : %d details: %lu", __FILE__, __LINE__, size*sizeof(unsigned char));
        return MEMORY_ERROR;
    }
    if (fread(byte_board, sizeof(unsigned char), size, in_file) != sizeof(unsigned char) * size)
    {
        fclose(in_file);
        free(byte_board);
        sprintf(log_message, "%s : %d details: %lu", __FILE__, __LINE__, sizeof(unsigned char)*size);
        return FREAD_MISSED;
    }
    fclose(in_file);

    cells = (unsigned char *)calloc((long unsigned)n_tails*n_tails*(height-n_tails+1)*(long unsigned)(width-n_tails+1), sizeof(unsigned char));
    if (cells == NULL)
    {
        free(byte_board);
        sprintf(log_message, "%s : %d details: %lu", __FILE__, __LINE__, ((long unsigned)n_tails*n_tails*(height-n_tails+1)*(long unsigned)(width-n_tails+1))*sizeof(unsigned char));
        return MEMORY_ERROR;
    }

    _synthetizer(cells, byte_board, width, height, n_tails);

    FILE* out_file = fopen(out_path, "wb");
    if (out_file == NULL)
    {
        free(byte_board);
        free(cells);
        sprintf(log_message, "%s : %d details: %s", __FILE__, __LINE__, out_path);
        return FOPEN_MISSED;
    }
    if (fwrite(
            cells,
            sizeof(unsigned char),
            ((long unsigned)n_tails*n_tails*(height-n_tails+1)*(long unsigned)(width-n_tails+1)),
            out_file
        ) != sizeof(unsigned char) * ((long unsigned)n_tails*n_tails*(height-n_tails+1)*(long unsigned)(width-n_tails+1)))
    {
        fclose(out_file);
        free(byte_board);
        free(cells);
        sprintf(log_message, "%s : %d details: %ld", __FILE__, __LINE__, sizeof(unsigned char) * ((long unsigned)n_tails*n_tails*(height-n_tails+1)*(long unsigned)(width-n_tails+1)));
        return FWRITE_MISSED;
    }

    fclose(out_file);
    free(byte_board);
    free(cells);

    return SUCCESS;
}


int csynthesis(
    const char * const in_db_path,
    const char * const out_db_path,
    const char * const list_file_path,
    const char * const log_file_path,
    unsigned n_tails,
    unsigned n_threads)
{
    /**
     * @note in_db_path and out_db_path exist
     * @note list_file_path is a file with paths of files relative to in_db_path
     * @note it's possible to create files in out_db_path with the same relative path
     * @note log_file_path is a file where to write the log, with following format
     * "%d %d %s", thread_num, status, file_path
     */

    FILE* list_file = fopen(list_file_path, "r");
    if (list_file == NULL)
    {
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
            strcpy(in_path, in_db_path);
            strcat(in_path, "/");
            strcat(in_path, path);
            strcpy(out_path, out_db_path);
            strcat(out_path, "/");
            strcat(out_path, path);
            int ret = _reader(in_path, out_path, log_message, n_tails);
            #pragma omp critical
            {
                if (ret != SUCCESS)
                {
                    FILE* log_file = fopen(log_file_path, "a");
                    if (log_file == NULL)
                    {
                        error_detected = SUPER_ERROR;
                    }
                    else
                    {
                        int chars_written = fprintf(log_file, "%d in %s\n", ret, log_message);
                        if (chars_written < 0)
                        {
                            error_detected = SUPER_ERROR;
                        }
                        else
                        {
                            error_detected = ret;
                        }
                        fclose(log_file);
                    }
                }
                else
                {
                    FILE* log_file = fopen(log_file_path, "a");
                    if (log_file == NULL)
                    {
                        error_detected = LOG_FOPEN_MISSED;
                    }
                    else
                    {
                        int chars_written = fprintf(log_file, "%d %s\n", omp_get_thread_num(), path);
                        if (chars_written < 0)
                        {
                            error_detected = LOG_FWRITE_MISSED;
                        }
                        fclose(log_file);
                    }
                }
            }
        }
    }
    if (error_detected != SUCCESS)
    {
        fclose(list_file);
        return error_detected;
    }

    return SUCCESS;
}
