#include "module.h"

PyDoc_STRVAR(test_nested_sequences_docstring, "test_nested_sequences("
                                              "b1: bool, t1: tuple[int, bool], t2: tuple[float, float, object], t3: "
                                              "tuple[int, tuple[str, bool, tuple[float, int]], str]"
                                              ") -> tuple[bool, tuple[int, bool], tuple[float, float, object], "
                                              "tuple[int, tuple[str, bool, tuple[float, int]], str]]\n"
                                              "Returns its arguments, which are parsed as deeply nested sequences.\n");

// Check that function with the following parameters can be parsed:
// bool, (int, bool), (float, float, PyObject), (int, (str, bool, (float, int)), str)
//   b1    i1    b2       f1     f2        o1     i2    s1    b3      f3   i3     s2
static PyObject *test_nested_sequences(const PyObject *const Py_UNUSED(mod), PyObject *const *const args,
                                       const Py_ssize_t nargs, const PyObject *const kwnames)
{
    int b1, b2, b3;
    Py_ssize_t i1, i2, i3;
    double f1, f2, f3;
    PyObject *o1;
    const char *s1, *s2;

    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_BOOL, .p_val = &b1},
                {.type = CPYARG_TYPE_SEQUENCE,
                 .p_val =
                     (cpyutl_argument_t[]){
                         {.type = CPYARG_TYPE_SSIZE, .p_val = &i1},
                         {.type = CPYARG_TYPE_BOOL, .p_val = &b2},
                         {},
                     }},
                {.type = CPYARG_TYPE_SEQUENCE,
                 .p_val =
                     (cpyutl_argument_t[]){
                         {.type = CPYARG_TYPE_DOUBLE, .p_val = &f1},
                         {.type = CPYARG_TYPE_DOUBLE, .p_val = &f2},
                         {.type = CPYARG_TYPE_PYTHON, .p_val = &o1},
                         {},
                     }},
                {.type = CPYARG_TYPE_SEQUENCE,
                 .p_val =
                     (cpyutl_argument_t[]){
                         {.type = CPYARG_TYPE_SSIZE, .p_val = &i2},
                         {.type = CPYARG_TYPE_SEQUENCE,
                          .p_val =
                              (cpyutl_argument_t[]){
                                  {.type = CPYARG_TYPE_STRING, .p_val = &s1},
                                  {.type = CPYARG_TYPE_BOOL, .p_val = &b3},
                                  {.type = CPYARG_TYPE_SEQUENCE,
                                   .p_val =
                                       (cpyutl_argument_t[]){
                                           {.type = CPYARG_TYPE_DOUBLE, .p_val = &f3},
                                           {.type = CPYARG_TYPE_SSIZE, .p_val = &i3},
                                           {},
                                       }},
                                  {},
                              }},
                         {.type = CPYARG_TYPE_STRING, .p_val = &s2},
                         {},
                     }},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    // Build the output value the same way the input is:
    // bool, (int, bool), (float, float, PyObject), (int, (str, bool, (float, int)), str)
    //   b1    i1    b2       f1     f2        o1     i2    s1    b3      f3   i3     s2

    return cpyutl_output_create_check(CPYOUT_TYPE_TUPLE,
                                      (cpyutl_output_t[]){
                                          {.type = CPYOUT_TYPE_PYBOOL, .value_bool = b1},
                                          {.type = CPYOUT_TYPE_TUPLE,
                                           .value_nested =
                                               (cpyutl_output_t[]){
                                                   {.type = CPYOUT_TYPE_PYINT, .value_int = i1},
                                                   {.type = CPYOUT_TYPE_PYBOOL, .value_bool = b2},
                                                   {},
                                               }},
                                          {.type = CPYOUT_TYPE_TUPLE,
                                           .value_nested =
                                               (cpyutl_output_t[]){
                                                   {.type = CPYOUT_TYPE_PYFLOAT, .value_float = f1},
                                                   {.type = CPYOUT_TYPE_PYFLOAT, .value_float = f2},
                                                   {.type = CPYOUT_TYPE_PYOBJ, .value_obj = o1},
                                                   {},
                                               }},
                                          {.type = CPYOUT_TYPE_TUPLE,
                                           .value_nested =
                                               (cpyutl_output_t[]){
                                                   {.type = CPYOUT_TYPE_PYINT, .value_int = i2},
                                                   {.type = CPYOUT_TYPE_TUPLE,
                                                    .value_nested =
                                                        (cpyutl_output_t[]){
                                                            {.type = CPYOUT_TYPE_PYSTRING, .value_str = s1},
                                                            {.type = CPYOUT_TYPE_PYBOOL, .value_bool = b3},
                                                            {.type = CPYOUT_TYPE_TUPLE,
                                                             .value_nested =
                                                                 (cpyutl_output_t[]){
                                                                     {.type = CPYOUT_TYPE_PYFLOAT, .value_float = f3},
                                                                     {.type = CPYOUT_TYPE_PYINT, .value_int = i3},
                                                                     {},
                                                                 }},
                                                            {},
                                                        }},
                                                   {.type = CPYOUT_TYPE_PYSTRING, .value_str = s2},
                                                   {},
                                               }},
                                          {},
                                      });
}

static PyModuleDef cpyutl_test_module = {
    PyModuleDef_HEAD_INIT,
    "cpyutl._cpyutl_test",
    PyDoc_STR("Test module for cpyutl."),
    0,
    (PyMethodDef[]){
        {
            .ml_name = "test_nested_sequences",
            .ml_meth = (void *)test_nested_sequences,
            .ml_flags = METH_FASTCALL | METH_KEYWORDS,
            .ml_doc = test_nested_sequences_docstring,
        },
        {},
    },
    (PyModuleDef_Slot[]){
        {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_SUPPORTED},
        {},
    },
};

PyMODINIT_FUNC PyInit__cpyutl_test(void)
{
    if (PyArray_ImportNumPyAPI() < 0)
        return NULL;

    return PyModuleDef_Init(&cpyutl_test_module);
}
