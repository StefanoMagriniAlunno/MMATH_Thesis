/**
 * @file main.c
 * @author Stefano Magrini Alunno (stefanomagrini99@gmail.com)
 * @brief main source file of the synthesis project. A python script call functions of this file.
 *
 * @version 0.0.0
 * @date 2024-07-16
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <Python.h>
#include <stdlib.h>
#include "synthesis.h"

/**
 * @brief This function read input from Python and call sum functions from synthesis.h
 *
 * @param[in] self
 * @param[in] args :
 * - (str) complete path of input dataset
 * - (str) complete path of output dataset
 * - (str) complete path of file with list of all files
 * - (str) complete path of log file
 * @return PyObject*
 *
 * @note This script recreate all files but synthetised with extension .synth
 * @note The file contains the relative paths of input files
 */
static PyObject* wrapper(PyObject* self, PyObject* args)
{
    const char *in_dset_path = NULL, *out_dset_path = NULL, *file_path = NULL, *log_path = NULL;
    int n_threads = 0;
    if (!PyArg_ParseTuple(args, "ssssi", &in_dset_path, &out_dset_path, &file_path, &log_path, &n_threads))
    {
        PyErr_SetString(PyExc_SyntaxError, "Required 3 strings in input and an integer.");
        return NULL;
    }
    if (! (n_threads > 0))
    {
        PyErr_Format(PyExc_ValueError, "Number of threads is not valid. Passed %d", n_threads);
        return NULL;
    }

    int ret = csynthesis(in_dset_path, out_dset_path, file_path, log_path, n_threads);

    switch(ret)
    {
        case LOG_FOPEN_MISSED:
            PyErr_SetString(PyExc_IOError, "Log file opening error");
            return NULL;
        case FOPEN_MISSED:
            PyErr_SetString(PyExc_IOError, "file opening error, see log file for more details...");
            return NULL;
        case FWRITE_MISSED:
            PyErr_SetString(PyExc_IOError, "file writing error, see log file for more details...");
            return NULL;
        case FREAD_MISSED:
            PyErr_SetString(PyExc_IOError, "file reading error, see log file for more details...");
            return NULL;
        case MEMORY_ERROR:
            PyErr_SetString(PyExc_MemoryError, "malloc or calloc failed the allocation...");
            return NULL;
        case SUPER_ERROR:
            PyErr_SetString(PyExc_Exception, "An error occurred and failed the report in log file.");
            return NULL;
    }

    // ritorno NONE
    Py_RETURN_NONE;
}

// 'methods' is the list of methods of the module
static PyMethodDef methods[] = {
    {"wrapper", wrapper, METH_VARARGS,
    "Call synthesis function\n\n"
    "This function reads the list of images, synthesises each image and returns the path of the synthesised images.\n\n"
    ":emphasis:`params`\n"
    "  - :attr:`self`: Reference to the module or object calling the method\n"
    "  - :attr:`args`: \n"
    "    - :type:`str`: complete path of input dataset\n"
    "    - :type:`str`: complete path of output dataset\n"
    "    - :type:`str`: complete path of file with all relative paths\n"
    "    - :type:`str`: complete path of log file\n"
    ":emphasis:`raises`\n"
    "  - :exc:`SyntaxError`\n"
    "  - :exc:`ValueError`\n"
    "  - :exc:`IOError`\n"
    "  - :exc:`MemoryError`\n"
    ":emphasis:`usage`\n"
    "  >>> import synthesis\n"
    "  >>> synthesis.wrapper('/path/to/in/db', '/path/to/out/db', '/path/to/listfile', '/path/to/logfile')\n"
    },
    {NULL, NULL, 0, NULL}  // Sentinel
};

// 'synthesis' is the module name
static struct PyModuleDef synthesis = {
    PyModuleDef_HEAD_INIT,
    "synthesis",      // Nome del modulo Python
    NULL,             // Documentazione del modulo
    -1,               // Permette il caricamento di più istanze dello stesso modulo in uno stesso spazio dei nomi
    methods           // Lista dei metodi del modulo
};

// Initialize the module
PyMODINIT_FUNC PyInit_synthesis(void)
{
    return PyModule_Create(&synthesis);
}
