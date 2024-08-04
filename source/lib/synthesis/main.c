/**
 * @file main.c
 * @author Stefano Magrini Alunno (stefanomagrini99@gmail.com)
 * @brief main source file of the synthesis project. A python script call functions of this file.
 *
 * @date 2024-07-16
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <Python.h>
#include <stdlib.h>
#include "synthesis.h"

/**
 * @brief This function read input from Python and call csynthesis function from synthesis.h
 *
 * @param self
 * @param args :
 * - (str) complete path of input dataset
 * - (str) complete path of output dataset
 * - (str) complete path of file with list of all files
 * - (int) size of tiles (pass N if you want to use NxN tiles)
 * - (int) number of threads
 * - (str) complete path of log file
 * @return PyObject*: None
 *
 * @exception SyntaxError : if the number or format of arguments is not valid
 * @exception ValueError : if the size of tiles or number of threads is not valid
 */
static PyObject* wrapper(PyObject* self, PyObject* args)
{
    if (self == NULL)
    {
        PyErr_SetString(PyExc_Exception, "self is NULL");
        return NULL;
    }
    const char *in_dset_path = NULL, *out_dset_path = NULL, *file_path = NULL, *log_path = NULL;
    int n_threads = 0, n_tiles = 0;
    if (!PyArg_ParseTuple(args, "sssiis", &in_dset_path, &out_dset_path, &file_path, &n_tiles, &n_threads, &log_path))
    {
        PyErr_SetString(PyExc_SyntaxError, "Required 3 strings in input and an integer.");
        return NULL;
    }
    if (! (n_tiles > 0))
    {
        PyErr_Format(PyExc_ValueError, "Size of tiles is not valid. Passed %d", n_tiles);
        return NULL;
    }
    if (! (n_threads > 0))
    {
        PyErr_Format(PyExc_ValueError, "Number of threads is not valid. Passed %d", n_threads);
        return NULL;
    }

    int ret = csynthesis(in_dset_path, out_dset_path, file_path, n_tiles, n_threads, log_path);

    switch(ret)
    {
        case LOG_ERROR:
            PyErr_Format(PyExc_Exception, "log file error, please check %s for more details", log_path);
            Py_RETURN_NONE;
        case IO_ERROR:
            PyErr_Format(PyExc_OSError, "an error occurred during IO operations, please check %s for more details", log_path);
            Py_RETURN_NONE;
        case VALUE_ERROR:
            PyErr_Format(PyExc_ValueError, "invalid input values, please check %s for more details", log_path);
            Py_RETURN_NONE;
        case MEMORY_ERROR:
            PyErr_Format(PyExc_MemoryError, "memory allocation error, please check %s for more details", log_path);
            Py_RETURN_NONE;
    }

    // ritorno NONE
    Py_RETURN_NONE;
}

// 'methods' is the list of methods of the module
static PyMethodDef methods[] = {
    {"wrapper", wrapper, METH_VARARGS,
    "Call synthesis function\n\n"
    "This function reads the list of images, synthesises each image.\n"
    ""
    "\n"
    ":param self: Reference to the module or object calling the method\n"
    ":type self: PyObject\n"
    ":param args: arguments:\n"
    "- (str) complete path of input dataset\n"
    "- (str) complete path of output dataset\n"
    "- (str) complete path of file with all relative paths\n"
    "- (int) size of tailse (the length of side)\n"
    "- (int) num of dedicated threads\n"
    "- (str) complete path of log file\n"
    ":raises SyntaxError:\n"
    ":raises ValueError:\n"
    ":raises OSError:\n"
    ":raises MemoryError:\n"
    ":raises Exception:\n"
    ":usage:\n"
    ">>> import synthesis\n"
    ">>> synthesis.wrapper('/path/to/in/db', '/path/to/out/db', '/path/to/listfile', 6, 8, '/path/to/logfile')\n"
    },
    {NULL, NULL, 0, NULL}  // Sentinel
};

// 'libsynthesis' is the module name
static struct PyModuleDef libsynthesis = {
    PyModuleDef_HEAD_INIT,
    "libsynthesis",   // Nome del modulo Python
    NULL,             // Documentazione del modulo
    -1,               // Permette il caricamento di più istanze dello stesso modulo in uno stesso spazio dei nomi
    methods,          // Lista dei metodi del modulo
    NULL,             // m_slots
    NULL,             // m_traverse
    NULL,             // m_clear
    NULL              // m_free
};

// Initialize the module
PyMODINIT_FUNC PyInit_libsynthesis(void)
{
    return PyModule_Create(&libsynthesis);
}
