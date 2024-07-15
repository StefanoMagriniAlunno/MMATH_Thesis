#include "sum.h"
#include <Python.h>

// Funzione di esempio che utilizza sum.h
static PyObject* example_function(PyObject* self, PyObject* args) {
    int a = 10;
    int b = 20;
    int result = sum(a, b);
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

// Funzione di inizializzazione del modulo
PyMODINIT_FUNC PyInit_example(void) {
    return PyModule_Create(&example_module);
}
