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
#include "synthesis.h"

/**
 * @brief This function read input from Python and call sum functions from synthesis.h
 *
 * @param[in] self
 * @param[in] args
 * @return PyObject*
 */
static PyObject* wrapper(PyObject* self, PyObject* args)
{
    csynthesis();
}

// 'methods' is the list of methods of the module
static PyMethodDef methods[] = {
    {"wrapper", wrapper, METH_NOARGS,
     "Call synthesis function\n\n"
     "This function reads the list of images, synthesises each image and returns the path of the synthesised images.\n\n"
     ":emphasis:`params`\n"
     "  - :attr:`self`: Reference to the module or object calling the method\n"
     "  - :attr:`args` :type:`List[str]`: List of paths of the images to be synthesised\n"
     ":emphasis:`return`\n"
     "  - :type:`str` path of the synthesised images.\n"
     ":emphasis:`raises`\n"
     "  - :exc:`TypeError` If the input is not a list of strings\n"
     "  - :exc:`Warning` If the input is an empty list\n"
     "  - :exc:`FileNotFoundError` If the input list contains a path that does not exist\n"
     "  - :exc:`ValueError` If the input list contains a path that is not a valid image\n"
     "  - :exc:`Exception` If an error occurs during the synthesis\n"
     ":emphasis:`usage`\n"
     "  >>> import synthesis\n"
     "  >>> ret = synthesis.wrapper(['path/to/image1.jpg', 'path/to/image2.jpg'])\n"
     "  >>> print(ret)\n"
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
