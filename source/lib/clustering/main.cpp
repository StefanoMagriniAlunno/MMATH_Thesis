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
 * @param self
 * @param args :
 * - (str) complete path of datafile
 * - (str) complete path of file with initial centroids
 * - (str) complete path of output file
 * - (int) dimension of data points
 * - (float) tollerance
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
    int dimension = 0;
    float tollerance = 0.0;

    if (!PyArg_ParseTuple(args, "sssifs", &datafile_path, &centroids_path, &outfile_path, &dimension, &tollerance, &log_path))
    {
        PyErr_SetString(PyExc_SyntaxError, "Expected 4 strings and 1 integer.");
        return NULL;
    }

    if (dimension <= 0)
    {
        PyErr_Format(PyExc_ValueError, "Size of data points is not valid. Passed %d", dimension);
        return NULL;
    }

    int ret = cxxfcm(datafile_path, outfile_path, centroids_path, dimension, tollerance, log_path);

    switch(ret)
    {
        case LOG_ERROR:
            PyErr_Format(PyExc_Exception, "log file error, please check %s for more details", log_path);
            Py_RETURN_NONE;
        case IO_ERROR:
            PyErr_Format(PyExc_OSError, "an error occurred during IO operations, please check %s for more details", log_path);
            Py_RETURN_NONE;
        case DEVICE_ERROR:
            PyErr_Format(PyExc_RuntimeError, "an error occurred during device operations, please check %s for more details", log_path);
            Py_RETURN_NONE;
    }

    // Return None
    Py_RETURN_NONE;
}

// 'methods' is the list of methods of the module
static PyMethodDef methods[] = {
    {"fcmwrapper", fcmwrapper, METH_VARARGS,
    "Perform Fuzzy C-Means Clustering\n"
    "This function reads the list of images, synthesises each image.\n"
    ""
    "\n"
    ":param self: Reference to the module or object calling the method\n"
    ":type self: PyObject\n"
    ":param args: arguments:\n"
    "- (str) complete path of datafile\n"
    "- (str) complete path of file with initial centroids\n"
    "- (str) complete path of output file\n"
    "- (int) dimension of data points\n"
    "- (float) tollerance\n"
    ":raises SyntaxError:\n"
    ":raises ValueError:\n"
    ":raises IOError:\n"
    ":raises RuntimeError:\n"
    ":raises Exception:\n"
    ":usage:\n"
    ">>> import synthesis\n"
    ">>> synthesis.wrapper('/path/to/in/db', '/path/to/out/db', '/path/to/listfile', 6, 8, '/path/to/logfile')\n"
    },
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
