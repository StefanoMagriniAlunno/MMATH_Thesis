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
 * @param[in] n_tiles: size of tiles
 * @param[in] n_threads: number of threads
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

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads)
    #endif  /* _OPENMP */
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
 * @param[in] in_path : complete path of the image
 * @param[in] out_path : complete path of the output
 * @param[out] log_message : message for log when there is an error
 * @param[in] n_tiles : size of tiles
 * @param[in] n_threads : num of threads
 * @return int: status code
 *
 */
int _reader(
    const char *const in_path,
    const char *const out_path,
    char log_message[MESSAGE_MAXLEN],
    const unsigned n_tiles,
    const unsigned n_threads)
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
    if (width < n_tiles && height < n_tiles)
    {
        fclose(in_file);
        sprintf(log_message, "%s : %d details: w=%u, h=%u, n_tiles=%u", __FILE__, __LINE__, width, height, n_tiles);
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

    cells = (unsigned char *)calloc((long unsigned)n_tiles*n_tiles*(height-n_tiles+1)*(long unsigned)(width-n_tiles+1), sizeof(unsigned char));
    if (cells == NULL)
    {
        free(byte_board);
        sprintf(log_message, "%s : %d details: %lu", __FILE__, __LINE__, ((long unsigned)n_tiles*n_tiles*(height-n_tiles+1)*(long unsigned)(width-n_tiles+1))*sizeof(unsigned char));
        return MEMORY_ERROR;
    }

    _synthetizer(cells, byte_board, width, height, n_tiles, n_threads);

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
            ((long unsigned)n_tiles*n_tiles*(height-n_tiles+1)*(long unsigned)(width-n_tiles+1)),
            out_file
        ) != sizeof(unsigned char) * ((long unsigned)n_tiles*n_tiles*(height-n_tiles+1)*(long unsigned)(width-n_tiles+1)))
    {
        fclose(out_file);
        free(byte_board);
        free(cells);
        sprintf(log_message, "%s : %d details: %ld", __FILE__, __LINE__, sizeof(unsigned char) * ((long unsigned)n_tiles*n_tiles*(height-n_tiles+1)*(long unsigned)(width-n_tiles+1)));
        return FWRITE_MISSED;
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
    const char *const log_file_path,
    const unsigned n_tiles,
    const unsigned n_threads)
{
    FILE* list_file = fopen(list_file_path, "r");
    if (list_file == NULL)
    {
        return FOPEN_MISSED;
    }

    int error_detected = SUCCESS;
    {
        char path[PATH_MAX];
        while (fgets(path, sizeof(path), list_file) != NULL && error_detected == SUCCESS)
        {
            path[strlen(path)-1] = '\0';  // last char is '\n'
            char log_message[MESSAGE_MAXLEN];
            char in_path[PATH_MAX];
            char out_path[PATH_MAX];
            strcpy(in_path, in_db_path);
            strcat(in_path, "/");
            strcat(in_path, path);
            strcpy(out_path, out_db_path);
            strcat(out_path, "/");
            strcat(out_path, path);
            int ret = _reader(in_path, out_path, log_message, n_tiles, n_threads);
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
                    int chars_written = fprintf(log_file, "%s processed\n", path);
                    if (chars_written < 0)
                    {
                        error_detected = LOG_FWRITE_MISSED;
                    }
                    fclose(log_file);
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
