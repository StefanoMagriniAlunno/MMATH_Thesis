/**
 * @file main.c
 * @author Stefano Magrini Alunno (stefanomagrini99@gmail.com)
 * @brief
 * @version 0.0.0
 * @date 2024-07-15
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <Python.h>
#include "sum.h"

/**
 * @brief Sum between two int of Python
 *
 * @param self (PyObject*) : reference to the module
 * @param args (PyObject*) : arguments passed to the function
 * @return PyObject* : sum between two int of Python
 */
static PyObject* example_function(PyObject* self, PyObject* args) {
    int32_t a, b;
    if (!PyArg_ParseTuple(args, "ii", &a, &b)) {
        return NULL;
    }
    int32_t result = sum(a, b);
    return PyLong_FromLong(result);
}

// Lista dei metodi disponibili nel modulo Python
static PyMethodDef example_methods[] = {
    {"example_function", example_function, METH_NOARGS, "Esempio di funzione Python che usa sum.h"},
    {NULL, NULL, 0, NULL}  // Fine della lista dei metodi
};

// Definizione del modulo Python
static struct PyModuleDef example_module = {
    PyModuleDef_HEAD_INIT,
    "example",      // Nome del modulo Python
    NULL,           // Documentazione del modulo
    -1,             // Permette il caricamento di più istanze dello stesso modulo in uno stesso spazio dei nomi
    example_methods // Lista dei metodi del modulo
};

/**
 * @brief Init function of the module
 *
 * @return PyMODINIT_FUNC : module
 */
PyMODINIT_FUNC PyInit_example(void) {
    return PyModule_Create(&example_module);
}
