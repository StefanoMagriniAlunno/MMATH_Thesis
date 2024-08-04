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
 * @param cells: array of double
 * @param byte_board: array of unsigned char
 * @param width: width of image
 * @param height: height of image
 * @param n_tiles: size of tiles
 * @param n_threads: number of threads
 */
void _synthetizer(
    unsigned char *const cells,
    const unsigned char *const byte_board,
    const unsigned width,
    const unsigned height,
    const unsigned n_tiles,
    const unsigned n_threads)
{
    unsigned width_short = width - n_tiles + 1, height_short = height - n_tiles + 1;

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (unsigned row = 0; row < height_short; ++row) for (unsigned col = 0; col < width_short; ++col)  // fisso il pixel che voglio inserire
        for (unsigned i = 0; i < n_tiles; ++i) for (unsigned j = 0; j < n_tiles; ++j)  // identifico dove inserirlo
        {
            long unsigned index_out = ((row*width_short + col)*n_tiles + i)*n_tiles + j;  // [row-i, col-j, i,j]
            long unsigned index_in = (row+i)*width + col+j;
            cells[index_out] = byte_board[index_in];
        }
}

/**
 * @brief This support function read an image and call the synthetizer.
 *
 * @param in_path : complete path of the image
 * @param out_path : complete path of the output
 * @param n_tiles : size of tiles
 * @param n_threads : num of threads
 * @param log_file : log file
 * @return int: status code
 *
 */
int _reader(
    const char *const in_path,
    const char *const out_path,
    const unsigned n_tiles,
    const unsigned n_threads,
    FILE* log_file)
{
    unsigned width = 0;
    unsigned height = 0;
    long unsigned size = 0;
    long unsigned cells_size = 0;
    unsigned char * cells = NULL;
    unsigned char * byte_board = NULL;
    FILE* in_file = fopen(in_path, "r");
    if (in_file == NULL)
    {
        fprintf(log_file, "in file %s at line %d : CRITICAL fopen error, input_path=%s\n", __FILE__, __LINE__, in_path);
        fflush(log_file);
        return IO_ERROR;
    }
    if (fscanf(in_file, "P5\n%u %u\n255", &width, &height) != 2)
    {
        fclose(in_file);
        fprintf(log_file, "in file %s at line %d : CRITICAL format error\n", __FILE__, __LINE__);
        fflush(log_file);
        return IO_ERROR;
    }
    fprintf(log_file, "in file %s at line %d : DEBUG width=%u, height=%u\n", __FILE__, __LINE__, width, height);
    if (width < n_tiles && height < n_tiles)
    {
        fclose(in_file);
        fprintf(log_file, "in file %s at line %d : CRITICAL value error, n_tiles=%u\n", __FILE__, __LINE__, n_tiles);
        fflush(log_file);
        return VALUE_ERROR;
    }
    fgetc(in_file);  // used to skip the character newline
    size = (unsigned long)width * (unsigned long)height;
    byte_board = (unsigned char *)calloc(size, sizeof(unsigned char));
    if (byte_board == NULL)
    {
        fclose(in_file);
        fprintf(log_file, "in file %s at line %d : CRITICAL calloc error, size=%lu\n", __FILE__, __LINE__, size);
        fflush(log_file);
        return MEMORY_ERROR;
    }
    if (fread(byte_board, sizeof(unsigned char), size, in_file) != sizeof(unsigned char) * size)
    {
        fclose(in_file);
        free(byte_board);
        fprintf(log_file, "in file %s at line %d : CRITICAL fread error\n", __FILE__, __LINE__);
        fflush(log_file);
        return IO_ERROR;
    }
    fclose(in_file);

    cells_size = (long unsigned)n_tiles*n_tiles*(height-n_tiles+1)*(width-n_tiles+1);
    cells = (unsigned char *)calloc(cells_size, sizeof(unsigned char));
    if (cells == NULL)
    {
        free(byte_board);
        fprintf(log_file, "in file %s at line %d : CRITICAL calloc error, cells_size=%lu\n", __FILE__, __LINE__, cells_size*sizeof(unsigned char));
        fflush(log_file);
        return MEMORY_ERROR;
    }

    fprintf(log_file, "in file %s at line %d : DEBUG starting synthetizer with %u threads\n", __FILE__, __LINE__, n_threads);
    fflush(log_file);
    _synthetizer(cells, byte_board, width, height, n_tiles, n_threads);

    FILE* out_file = fopen(out_path, "wb");
    if (out_file == NULL)
    {
        free(byte_board);
        free(cells);
        fprintf(log_file, "in file %s at line %d : CRITICAL fopen error, out_path=%s\n", __FILE__, __LINE__, out_path);
        fflush(log_file);
        return IO_ERROR;
    }
    if (fwrite(
            cells,
            sizeof(unsigned char),
            cells_size,
            out_file
        ) != sizeof(unsigned char) * cells_size)
    {
        fclose(out_file);
        free(byte_board);
        free(cells);
        fprintf(log_file, "in file %s at line %d : CRITICAL fwrite error\n", __FILE__, __LINE__);
        fflush(log_file);
        return IO_ERROR;
    }

    fclose(out_file);
    free(byte_board);
    free(cells);

    return SUCCESS;
}


int csynthesis(
    const char *const in_db_path,
    const char *const out_db_path,
    const char *const list_file_path,
    const unsigned n_tiles,
    const unsigned n_threads,
    const char *const log_file_path)
{
    FILE* list_file = fopen(list_file_path, "r");
    if (list_file == NULL)
    {
        return IO_ERROR;
    }
    FILE* log_file = fopen(log_file_path, "a");
    if (log_file == NULL)
    {
        fclose(list_file);
        return LOG_ERROR;
    }

    int error_detected = SUCCESS;
    {
        char path[PATH_MAX];
        while (fgets(path, sizeof(path), list_file) != NULL && error_detected == SUCCESS)
        {
            path[strlen(path)-1] = '\0';  // last char is '\n'
            char in_path[PATH_MAX];
            char out_path[PATH_MAX];
            strcpy(in_path, in_db_path);
            strcat(in_path, "/");
            strcat(in_path, path);
            strcpy(out_path, out_db_path);
            strcat(out_path, "/");
            strcat(out_path, path);
            {
                int chars_written = fprintf(log_file, "in file %s at line %d : INFO processing %s\n", __FILE__, __LINE__, path);
                if (chars_written < 0)
                {
                    fclose(list_file);
                    fclose(log_file);
                    return LOG_ERROR;
                }
                fflush(log_file);
            }
            int ret = _reader(in_path, out_path, n_tiles, n_threads, log_file);
            if (ret != SUCCESS)
            {
                int chars_written = fprintf(log_file, "in file %s at line %d : CRITICAL _reader error, code %d\n", __FILE__, __LINE__, ret);
                if (chars_written < 0)
                {
                    fclose(list_file);
                    fclose(log_file);
                    return LOG_ERROR;
                }
                else
                {
                    error_detected = ret;
                }
                fflush(log_file);
            }
        }
    }
    if (error_detected != SUCCESS)
    {
        fclose(list_file);
        fclose(log_file);
        return error_detected;
    }

    fclose(list_file);
    fclose(log_file);

    return SUCCESS;
}
