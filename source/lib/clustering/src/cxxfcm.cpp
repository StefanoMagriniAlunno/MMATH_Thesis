/**
 * @file cxxfcm.cpp
 * @author Stefano Magrini Alunno (stefanomagrini99@gmail.com)
 * @brief definition of cxxfcm
 *
 * @date 2024-07-16
 *
 * @copyright Copyright (c) 2024
 */
#include "fcm.h"
#include <fstream>
#include <string>
#include <vector>

// used to pass a log instruction
#define LOGGER(log_stream, level, message)                                    \
  do                                                                          \
    {                                                                         \
      log_stream << __FILE__ << ":" << __LINE__ << " \t" << (level) << " "    \
                 << (message) << std::endl;                                   \
      log_stream.flush ();                                                    \
    }                                                                         \
  while (0)

int
cxxfcm (const char *const datafile_path, const char *const weightsfile_path,
        const char *const centroids_path,
        const char *const outfile_centroids_path, size_t n_dimensions,
        float tollerance, const char *const log_path)
{
  // create log stream as new file
  std::ofstream log_stream (log_path, std::ios::trunc);
  if (!log_stream.is_open ())
    {
      return LOG_ERROR;
    }

  std::vector<float> data, weights, centroids;

  // read datafile_path
  {
    std::ifstream data_stream (datafile_path, std::ios::binary);
    if (!data_stream.is_open ())
      {
        LOGGER (log_stream, "CRITICAL",
                "is_open error, datafile_path=" + std::string (datafile_path));
        log_stream.close ();
        return IO_ERROR;
      }

    // Get data_stream size
    data_stream.seekg (0, std::ios::end);
    long unsigned size = data_stream.tellg ();
    data_stream.seekg (0, std::ios::beg);
    // Compute number of floats
    long unsigned n_floats = size / sizeof (float);

    // Read data
    data.resize (n_floats);
    data_stream.read (reinterpret_cast<char *> (data.data ()), size);
    data_stream.close ();
  }

  // read weights_path
  {
    std::ifstream weights_stream (weightsfile_path, std::ios::binary);
    if (!weights_stream.is_open ())
      {
        LOGGER (log_stream, "CRITICAL",
                "is_open error, weights_path=" + std::string (weightsfile_path));
        log_stream.close ();
        return IO_ERROR;
      }

    // Get weights_stream size
    weights_stream.seekg (0, std::ios::end);
    long unsigned size = weights_stream.tellg ();
    weights_stream.seekg (0, std::ios::beg);
    // Compute number of floats
    long unsigned n_floats = size / sizeof (float);

    // Read data
    weights.resize (n_floats);
    weights_stream.read (reinterpret_cast<char *> (weights.data ()), size);
    weights_stream.close ();
  }

  // read centroids_path
  {
    std::ifstream centroids_stream (centroids_path, std::ios::binary);
    if (!centroids_stream.is_open ())
      {
        LOGGER (log_stream, "CRITICAL",
                "is_open error, centroids_path="
                    + std::string (centroids_path));
        log_stream.close ();
        return IO_ERROR;
      }

    // Get centroids_stream size
    centroids_stream.seekg (0, std::ios::end);
    long unsigned size = centroids_stream.tellg ();
    centroids_stream.seekg (0, std::ios::beg);
    // Compute number of floats
    long unsigned n_floats = size / sizeof (float);

    // Read data
    centroids.resize (n_floats);
    centroids_stream.read (reinterpret_cast<char *> (centroids.data ()), size);
    centroids_stream.close ();
  }

  // check if data, weights have the same number of data points
  if (data.size () / n_dimensions != weights.size ())
    {
      LOGGER (log_stream, "CRITICAL", "data and weights have different size");
      log_stream.close ();
      return IO_ERROR;
    }

  // call CUDA function
  try
    {
      std::vector<float> out_centroids = cudafcm (
          data, weights, centroids, n_dimensions, tollerance, log_stream);

      // save out_centroids in outfile_centroids_path
      {
        std::ofstream outfile_stream (outfile_centroids_path,
                                      std::ios::binary);
        if (!outfile_stream.is_open ())
          {
            LOGGER (log_stream, "CRITICAL",
                    "is_open error, outfile_path="
                        + std::string (outfile_centroids_path));
            log_stream.close ();
            return IO_ERROR;
          }
        outfile_stream.write (reinterpret_cast<char *> (out_centroids.data ()),
                              out_centroids.size () * sizeof (float));
        outfile_stream.close ();
      }

      log_stream.close ();
    }
  catch (const std::bad_alloc &e)
    {
      LOGGER (log_stream, "CRITICAL", "bad_alloc caught");
      log_stream.close ();
      return DEVICE_ERROR;
    }
  catch (const std::runtime_error &e)
    {
      LOGGER (log_stream, "CRITICAL",
              "runtime_error caught " + std::string (e.what ()));
      log_stream.close ();
      return DEVICE_ERROR;
    }

  return SUCCESS;
}
