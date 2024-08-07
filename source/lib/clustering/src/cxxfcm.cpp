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

int
cxxfcm (const char *const datafile_path, const char *const outfile_path,
        const char *const centroids_path, size_t n_dimensions,
        float tollerance, const char *const log_path)
{
  // create log stream as new file
  std::ofstream log_stream (log_path, std::ios::trunc);
  if (!log_stream.is_open ())
    {
      return LOG_ERROR;
    }

  std::vector<float> data, initial_centroids;

  // read datafile_path
  {
    std::ifstream data_stream (datafile_path, std::ios::binary);
    if (!data_stream.is_open ())
      {
        log_stream << "in file " << __FILE__ << " at line " << __LINE__
                   << " : CRITICAL is_open error, datafile_path="
                   << datafile_path << std::endl;
        log_stream.flush ();
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

  // read centroids_path
  {
    std::ifstream centroids_stream (centroids_path, std::ios::binary);
    if (!centroids_stream.is_open ())
      {
        log_stream << "in file " << __FILE__ << " at line " << __LINE__
                   << " : CRITICAL is_open error, centroids_path="
                   << centroids_path << std::endl;
        log_stream.flush ();
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
    initial_centroids.resize (n_floats);
    centroids_stream.read (
        reinterpret_cast<char *> (initial_centroids.data ()), size);
    centroids_stream.close ();
  }

  // call CUDA function
  try
    {
      std::vector<float> centroids = cudafcm (
          data, initial_centroids, n_dimensions, tollerance, log_stream);

      // save centroids in outfile_path
      {
        std::ofstream outfile_stream (outfile_path, std::ios::binary);
        if (!outfile_stream.is_open ())
          {
            log_stream << "in file " << __FILE__ << " at line " << __LINE__
                       << " : CRITICAL is_open error, outfile_path="
                       << outfile_path << std::endl;
            log_stream.flush ();
            log_stream.close ();
            return IO_ERROR;
          }
        outfile_stream.write (reinterpret_cast<char *> (centroids.data ()),
                              centroids.size () * sizeof (float));
        outfile_stream.close ();
      }
    }
  catch (const std::bad_alloc &e)
    {
      log_stream << "in file " << __FILE__ << " at line " << __LINE__
                 << " :  CRITICAL caught " << e.what () << std::endl;
      log_stream.flush ();
      log_stream.close ();
      return DEVICE_ERROR;
    }
  catch (const std::runtime_error &e)
    {
      log_stream << "in file " << __FILE__ << " at line " << __LINE__
                 << " :  CRITICAL caught " << e.what () << std::endl;
      log_stream.flush ();
      log_stream.close ();
      return DEVICE_ERROR;
    }

  return SUCCESS;
}
