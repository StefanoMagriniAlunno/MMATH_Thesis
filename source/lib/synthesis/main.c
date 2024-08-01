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
 * @param[in] self
 * @param[in] args :
 * - (str) complete path of input dataset
 * - (str) complete path of output dataset
 * - (str) complete path of file with list of all files
 * - (str) complete path of log file
 * - (int) tails' size (pass N if you want to use NxN tails)
 * - (int) number of threads
 * @return PyObject*: None
 *
 * @exception SyntaxError : if the number or format of arguments is not valid
 * @exception ValueError : if the size of tails or number of threads is not valid
 */
static PyObject* wrapper(PyObject* self, PyObject* args)
{
    if (self == NULL)
    {
        PyErr_SetString(PyExc_Exception, "self is NULL");
        return NULL;
    }
    const char *in_dset_path = NULL, *out_dset_path = NULL, *file_path = NULL, *log_path = NULL;
    int n_threads = 0, n_tails = 0;
    if (!PyArg_ParseTuple(args, "ssssii", &in_dset_path, &out_dset_path, &file_path, &log_path, &n_tails, &n_threads))
    {
        PyErr_SetString(PyExc_SyntaxError, "Required 3 strings in input and an integer.");
        return NULL;
    }
    if (! (n_tails > 0))
    {
        PyErr_Format(PyExc_ValueError, "Size of tails is not valid. Passed %d", n_tails);
        return NULL;
    }
    if (! (n_threads > 0))
    {
        PyErr_Format(PyExc_ValueError, "Number of threads is not valid. Passed %d", n_threads);
        return NULL;
    }

    int ret = csynthesis(in_dset_path, out_dset_path, file_path, log_path, n_tails, n_threads);

    switch(ret)
    {
        case LOG_FOPEN_MISSED:
            PyErr_SetString(PyExc_IOError, "Log file opening error");
            Py_RETURN_NONE;
        case FOPEN_MISSED:
            PyErr_SetString(PyExc_IOError, "file opening error, see log file for more details...");
            Py_RETURN_NONE;
        case FWRITE_MISSED:
            PyErr_SetString(PyExc_IOError, "file writing error, see log file for more details...");
            Py_RETURN_NONE;
        case FREAD_MISSED:
            PyErr_SetString(PyExc_IOError, "file reading error, see log file for more details...");
            Py_RETURN_NONE;
        case MEMORY_ERROR:
            PyErr_SetString(PyExc_MemoryError, "malloc or calloc failed the allocation...");
            Py_RETURN_NONE;
        case VALUE_ERROR:
            PyErr_SetString(PyExc_ValueError, "uncorrect size of tails...");
            Py_RETURN_NONE;
        case SUPER_ERROR:
            PyErr_SetString(PyExc_Exception, "An error occurred and failed the report in log file.");
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
    ":emphasis:`params`\n"
    "  - :attr:`self`: Reference to the module or object calling the method\n"
    "  - :attr:`args`: \n"
    "    - :type:`str`: complete path of input dataset\n"
    "    - :type:`str`: complete path of output dataset\n"
    "    - :type:`str`: complete path of file with all relative paths\n"
    "    - :type:`str`: complete path of log file\n"
    "    - :type:`int`: size of tailse (the length of side)\n"
    "    - :type:`int`: num of dedicated threads\n"
    ":emphasis:`raises`\n"
    "  - :exc:`SyntaxError`\n"
    "  - :exc:`ValueError`\n"
    "  - :exc:`IOError`\n"
    "  - :exc:`MemoryError`\n"
    "  - :exc:`Exception`\n"
    ":emphasis:`usage`\n"
    "  >>> import synthesis\n"
    "  >>> synthesis.wrapper('/path/to/in/db', '/path/to/out/db', '/path/to/listfile', '/path/to/logfile', 6, 8)\n"
    },
    {NULL, NULL, 0, NULL}  // Sentinel
};

// 'synthesis' is the module name
static struct PyModuleDef synthesis = {
    PyModuleDef_HEAD_INIT,
    "synthesis",      // Nome del modulo Python
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
    return PyModule_Create(&synthesis);
}
