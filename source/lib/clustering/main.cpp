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
 * @brief This function reads input from Python and calls the cxxfcm function from fcm.h
 *
 * @param[in] self
 * @param[in] args :
 * - (str) complete path of datafile
 * - (str) complete path of file with initial centroids
 * - (str) complete path of output file
 * - (int) tiles' size (pass N if you want to use NxN tiles)
 * - (str) complete path of log file
 * @return PyObject*: None
 *
 * @exception SyntaxError : if the number or format of arguments is not valid
 * @exception ValueError : if the size of tiles is not valid
 *
 * @note This function returns a None PyObject
 */
extern "C" {
PyObject* fcmwrapper(PyObject* self, PyObject* args)
{
    if (self == NULL)
    {
        PyErr_SetString(PyExc_Exception, "self is NULL");
        return NULL;
    }
    const char *datafile_path = NULL, *outfile_path = NULL, *centroids_path = NULL, *log_path = NULL;
    int n_tiles = 0;

    if (!PyArg_ParseTuple(args, "sssis", &datafile_path, &centroids_path, &outfile_path, &n_tiles, &log_path))
    {
        PyErr_SetString(PyExc_SyntaxError, "Expected 4 strings and 1 integer.");
        return NULL;
    }

    if (n_tiles <= 0)
    {
        PyErr_Format(PyExc_ValueError, "Size of tiles is not valid. Passed %d", n_tiles);
        return NULL;
    }

    cxxfcm(datafile_path, outfile_path, centroids_path, n_tiles, log_path);

    // Return None
    Py_RETURN_NONE;
}

// 'methods' is the list of methods of the module
static PyMethodDef methods[] = {
    {"fcmwrapper", fcmwrapper, METH_VARARGS, "Wrapper for the cxxfcm function"},
    {NULL, NULL, 0, NULL}  // Sentinel
};

// 'libclustering' is the module name
static struct PyModuleDef libclustering = {
    PyModuleDef_HEAD_INIT,
    "libclustering",  // Name of the Python module
    NULL,             // Module documentation
    -1,               // Module keeps state in global variables
    methods,          // Methods of the module
    NULL,             // m_slots
    NULL,             // m_traverse
    NULL,             // m_clear
    NULL              // m_free
};

// Initialize the module
PyMODINIT_FUNC PyInit_libclustering(void)
{
    return PyModule_Create(&libclustering);
}
}
