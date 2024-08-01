/**
 * @file main.cpp
 * @author Stefano Magrini Alunno (stefanomagrini99@gmail.com)
 * @brief main source file of the fcm project. A python script call functions of this file.
 *
 * @date 2024-07-31
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <Python.h>
#include "fcm.h"


/**
 * @brief This function read input from Python and call cxxfcm function from fcm.h
 *
 * @param[in] self
 * @param[in] args :
 * - (str) complete path of datafile
 * - (str) complete path of output file
 * - (str) complete path of log file
 * - (str) complete path of file with initial centroids
 * - (int) tails' size (pass N if you want to use NxN tails)
 * @return PyObject*: None
 *
 * @exception SyntaxError : if the number or format of arguments is not valid
 * @exception ValueError : if the size of tails or number of threads is not valid
 *
 * @note This function return a
 */
extern "C" PyObject* wrapper(PyObject* self, PyObject* args)
{
    if (self == NULL)
    {
        PyErr_SetString(PyExc_Exception, "self is NULL");
        return NULL;
    }
    const char *datafile_path = NULL, *outfile_path = NULL, *log_path = NULL, *centroids_path = NULL;
    int n_tails = 0, n_centroids = 0;
    if (!PyArg_ParseTuple(args, "ssssii", &datafile_path, &outfile_path, &log_path, &centroids_path, &n_centroids, &n_tails))
    {
        PyErr_SetString(PyExc_SyntaxError, "Required 4 strings in input and 2 integers.");
        return NULL;
    }
    if (! (n_tails > 0))
    {
        PyErr_Format(PyExc_ValueError, "Size of tails is not valid. Passed %d", n_tails);
        return NULL;
    }

    int ret = cxxfcm(datafile_path, outfile_path, log_path, centroids_path, n_centroids, n_tails);

    // ritorno None
    Py_RETURN_NONE;
}
