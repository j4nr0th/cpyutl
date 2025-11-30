#include "cpyutl.h"

[[gnu::format(printf, 2, 3)]]
void raise_exception_from_current(PyObject *exception, const char *format, ...)
{
    PyObject *const original = PyErr_GetRaisedException();
    if (original)
    {
        va_list args;
        va_start(args, format);
        PyObject *const message = PyUnicode_FromFormatV(format, args);
        va_end(args);
        PyObject *new_exception = NULL;
        if (message)
        {
            new_exception = PyObject_CallFunctionObjArgs(exception, message, NULL);
            Py_DECREF(message);
        }

        if (new_exception)
        {
            PyException_SetCause(new_exception, original);
            PyErr_SetRaisedException(new_exception);
            new_exception = NULL;
        }
        else
        {
            PyErr_SetRaisedException(original);
        }
    }
    else
    {
        va_list args;
        va_start(args, format);
        PyErr_FormatV(exception, format, args);
        va_end(args);
    }
}

static cpyutl_argument_status_t validate_arg_specs(const unsigned n, const cpyutl_argument_t specs[const static n],
                                                   const int is_in_sequence)
{
    // Validate input specs have all keyword args at the end
    for (unsigned i = 1; i < n; ++i)
    {
        if (specs[i - 1].kwname != NULL && specs[i].kwname == NULL)
        {
            fprintf(stderr, "Argument %u has a keyword argument, but argument %u does not (author is a retard).\n",
                    i - 1, i);
            return CPYARG_BAD_SPECS;
        }
    }
    for (unsigned i = 0; i < n; ++i)
    {
        // If in a sequence, keywords are forbidden
        if (is_in_sequence && specs[i].kwname != NULL)
        {
            fprintf(stderr, "Keywords may not be used within a sequence.\n");
            return CPYARG_KW_IN_SEQUENCE;
        }

        // Keyword-only needs a keyword
        if (specs[i].kwname == NULL && specs[i].kw_only)
        {
            fprintf(stderr,
                    "Argument %u was marked as keyword-only, but does not specify a keyword (author is a retard).\n",
                    i);
            return CPYARG_BAD_SPECS;
        }
        // Typechecking is only for Python arguments
        if ((specs[i].type_check != NULL && specs[i].type != CPYARG_TYPE_PYTHON) &&
            (specs[i].type != CPYARG_TYPE_CUSTOM && specs[i].custom_convert != NULL))
        {
            fprintf(stderr,
                    "Argument %u specifies a type to check or custom callback, but does not specify the type as Python "
                    "object or custom "
                    "(author is a retard).\n",
                    i);
            return CPYARG_BAD_SPECS;
        }
        // Keywords cannot be repeated or empty
        if (specs[i].kwname != NULL)
        {
            for (unsigned j = i + 1; j < n; ++j)
            {
                if (specs[j].kwname != NULL && strcmp(specs[j].kwname, specs[i].kwname) == 0)
                {
                    fprintf(stderr, "Arguments %u and %u use the same keyword \"%s\" (author is a retard).\n", i, j,
                            specs[i].kwname);
                    return CPYARG_BAD_SPECS;
                }
            }
            if (specs[i].kwname[0] == '\0')
            {
                fprintf(stderr, "Argument %u specifies a keyword with no length.\n", i);
                return CPYARG_BAD_SPECS;
            }
        }
        // Value pointer must be given.
        if (specs[i].p_val == NULL)
        {
            fprintf(stderr, "Argument %u has no value pointer.\n", i);
            return CPYARG_BAD_SPECS;
        }
        // Type must be valid
        switch (specs[i].type)
        {
        case CPYARG_TYPE_PYTHON:
        case CPYARG_TYPE_BOOL:
        case CPYARG_TYPE_SSIZE:
        case CPYARG_TYPE_DOUBLE:
        case CPYARG_TYPE_STRING:
        case CPYARG_TYPE_SEQUENCE:
            break;

        case CPYARG_TYPE_NONE:
        default:
            fprintf(stderr, "Argument %u has invalid type %u.", i, specs[i].type);
            return CPYARG_BAD_SPECS;
        }
        // Sequence is recursively parsed
        if (specs[i].type == CPYARG_TYPE_SEQUENCE)
        {
            const cpyutl_argument_t *const seq_arg = specs[i].p_val;
            unsigned m = 0;
            while (seq_arg[m].type != CPYARG_TYPE_NONE)
            {
                m += 1;
            }
            const cpyutl_argument_status_t res = validate_arg_specs(m, seq_arg, 1);
            if (res != CPYARG_SUCCESS)
            {
                fprintf(stderr, "Could not parse sequence of length %u in argument %u (\"%s\")\n", m, i,
                        specs[i].kwname ? specs[i].kwname : "");
                return res;
            }
        }
    }

    return CPYARG_SUCCESS;
}

static cpyutl_argument_status_t extract_argument_value(const unsigned i, PyObject *const val,
                                                       cpyutl_argument_t *const arg)
{
    switch (arg->type)
    {
    case CPYARG_TYPE_PYTHON:
        if (arg->type_check && !PyObject_TypeCheck(val, arg->type_check))
        {
            PyErr_Format(PyExc_TypeError, "Argument %u is not of type %s but instead %R.", i, arg->type_check->tp_name,
                         Py_TYPE(val));
            return CPYARG_INVALID;
        }
        *(PyObject **)arg->p_val = val;
        break;

    case CPYARG_TYPE_BOOL:
        *(int *)arg->p_val = PyObject_IsTrue(val);
        if (PyErr_Occurred())
            return CPYARG_INVALID;
        break;

    case CPYARG_TYPE_SSIZE:
        *(Py_ssize_t *)arg->p_val = PyNumber_AsSsize_t(val, PyExc_ValueError);
        if (PyErr_Occurred())
            return CPYARG_INVALID;
        break;

    case CPYARG_TYPE_DOUBLE:
        *(double *)arg->p_val = PyFloat_AsDouble(val);
        if (PyErr_Occurred())
            return CPYARG_INVALID;
        break;

    case CPYARG_TYPE_STRING:
        *(const char **)arg->p_val = PyUnicode_AsUTF8(val);
        if (PyErr_Occurred())
            return CPYARG_INVALID;
        break;

    case CPYARG_TYPE_SEQUENCE:
    {
        PyObject *const fast_seq = PySequence_Fast(val, "Expected a sequence.");
        if (!fast_seq)
            return CPYARG_INVALID;
        const Py_ssize_t n = PySequence_Fast_GET_SIZE(fast_seq);
        const cpyutl_argument_status_t status = parse_arguments(
            (cpyutl_argument_t *)arg->p_val, (PyObject *const *)PySequence_Fast_ITEMS(fast_seq), n, NULL);
        Py_DECREF(fast_seq);
        if (status != CPYARG_SUCCESS)
            return status;
    }
    break;

    case CPYARG_TYPE_CUSTOM:
        if (arg->custom_convert(val, arg->p_val, arg->kwname))
            return CPYARG_INVALID;
        break;

    case CPYARG_TYPE_NONE:
        CPYUTL_ASSERT(0, "Should not be reached.");
        return CPYARG_BAD_SPECS;
    }
    arg->found = 1;

    return CPYARG_SUCCESS;
}

CPYUTL_INTERNAL
cpyutl_argument_status_t parse_arguments(cpyutl_argument_t specs[const], PyObject *const args[const],
                                         const Py_ssize_t nargs, const PyObject *const kwnames)
{
    CPYUTL_ASSERT(args != NULL, "Pointer to positional args should not be null.");
    const unsigned nkwds = kwnames != NULL ? PyTuple_GET_SIZE(kwnames) : 0;
    CPYUTL_ASSERT(specs != NULL, "Pointer to argument specs should not be null.");
    CPYUTL_ASSERT(nargs >= 0, "Number of arguments must be non-negative (it was %lld).", (long long int)nargs);

    unsigned n = 0;
    while (specs[n].type != CPYARG_TYPE_NONE)
    {
        specs[n].found = 0;
        n += 1;
    }
    if (n == 0)
    {
        // No args? Not my problem!
        return CPYARG_SUCCESS;
    }

    // Validate the arguments are properly specified.
    (void)validate_arg_specs;
    CPYUTL_ASSERT(validate_arg_specs(n, specs, 0) == CPYARG_SUCCESS, "Invalid argument specs.");
    CPYUTL_ASSERT(
        nargs + nkwds <= n,
        "Number of specified arguments is less than the number of received arguments (n = %u, nargs = %u, nkwds = %u).",
        n, (unsigned)nargs, (unsigned)nkwds);

    for (unsigned i = 0; i < nargs; ++i)
    {
        PyObject *const val = args[i];
        cpyutl_argument_t *const arg = specs + i;
        if (arg->kw_only)
        {
            PyErr_Format(PyExc_RuntimeError, "Argument %u (%s) is keyword-only, but was passed as positional argument.",
                         i, arg->kwname);
            return CPYARG_KW_AS_POS;
        }

        const cpyutl_argument_status_t res = extract_argument_value(i, val, arg);
        if (res != CPYARG_SUCCESS)
            return res;
    }

    unsigned first_kw = 0;
    while (first_kw < n && specs[first_kw].kwname == NULL)
    {
        first_kw += 1;
    }

    for (unsigned i = 0; i < nkwds; ++i)
    {
        PyObject *const val = args[nargs + i];
        PyObject *const kwname = PyTuple_GET_ITEM(kwnames, i);

        const char *kwd = PyUnicode_AsUTF8(kwname);
        if (!kwd)
            return CPYARG_INVALID;

        unsigned i_arg;
        for (i_arg = first_kw; i_arg < n; ++i_arg)
        {
            if (strcmp(kwd, specs[i_arg].kwname) == 0)
                break;
        }

        if (i_arg == n)
        {
            PyErr_Format(PyExc_TypeError, "Function does not have any parameter names \"%s\".", kwd);
            return CPYARG_NO_KW;
        }

        cpyutl_argument_t *const arg = specs + i_arg;

        if (arg->found)
        {
            PyErr_Format(PyExc_TypeError, "Parameter \"%s\" was already specified.", kwd);
            return CPYARG_DUPLICATE;
        }

        const cpyutl_argument_status_t res = extract_argument_value(i, val, arg);
        if (res != CPYARG_SUCCESS)
            return res;
    }

    for (unsigned i = 0; i < n; ++i)
    {
        const cpyutl_argument_t *const arg = specs + i;
        if (arg->found == 0 && arg->optional == 0)
        {
            PyErr_Format(PyExc_TypeError, "Non-optional parameter \"%s\" was not specified.", arg->kwname);
            return CPYARG_MISSING;
        }
    }

    return CPYARG_SUCCESS;
}

const char *cpyutl_output_type_str(const cpyutl_output_type_t e)
{
    switch (e)
    {
    case CPYOUT_TYPE_DICT:
        return "PyDictObject";
    case CPYOUT_TYPE_LIST:
        return "PyListObject";
    case CPYOUT_TYPE_TUPLE:
        return "PyTupleObject";
    case CPYOUT_TYPE_NONE:
        return "NONE";
    case CPYOUT_TYPE_PYFLOAT:
        return "PyFloatObject";
    case CPYOUT_TYPE_PYINT:
        return "PyLongObject";
    case CPYOUT_TYPE_PYOBJ:
        return "PyObject";
    case CPYOUT_TYPE_PYSTRING:
        return "PyUnicodeObject";
    case CPYOUT_TYPE_PYBOOL:
        return "PyBoolObject";
    default:
        return "UNKNOWN";
    }
}
const char *cpyutl_output_status_str(const cpyutl_output_status_t e)
{
    switch (e)
    {
    case CPYOUT_SUCCESS:
        return "Built correctly";
    case CPYOUT_FAILED_CONSTRUCTION:
        return "Construction using CPython's functions failed";
    case CPYOUT_INVALID_SPEC:
        return "Output specification was invalid";
    case CPYOUT_FAILED_DICT_INSERTION:
        return "Failed to insert a value into a dictionary";
    case CPYOUT_INVALID_OUT_TYPE:
        return "Output type was invalid";
    case CPYOUT_INVALID_NO_NAME:
        return "Output name was not given for a dict entry";
    case CPYOUT_NO_NESTED:
        return "Nested sequence had not values";
    case CPYOUT_INVALID_VALUE:
        return "Invalid (or null) value was specified";
    default:
        return "Unknown error";
    }
}

static cpyutl_output_status_t output_create_tuple(const cpyutl_output_t outputs[], PyObject **out);
static cpyutl_output_status_t output_create_list(const cpyutl_output_t outputs[], PyObject **out);
static cpyutl_output_status_t output_create_dict(const cpyutl_output_t outputs[], PyObject **out);

static cpyutl_output_status_t output_create_value(const cpyutl_output_t *output, PyObject **p_out)
{
    PyObject *out = NULL;
    switch (output->type)
    {
    case CPYOUT_TYPE_PYFLOAT:
        out = PyFloat_FromDouble(output->value_float);
        break;
    case CPYOUT_TYPE_PYINT:
        out = PyLong_FromLongLong(output->value_int);
        break;
    case CPYOUT_TYPE_PYOBJ:
        out = output->value_obj;
        Py_INCREF(out);
        break;
    case CPYOUT_TYPE_PYSTRING:
        out = PyUnicode_FromString(output->value_str);
        break;
    case CPYOUT_TYPE_PYBOOL:
        out = PyBool_FromLong((long)output->value_bool);
        break;
    case CPYOUT_TYPE_NONE:
        CPYUTL_ASSERT(0, "Tried to call create function on NONE value");
        return CPYOUT_INVALID_SPEC;
    case CPYOUT_TYPE_DICT:
        return output_create_dict(output->value_nested, p_out);
    case CPYOUT_TYPE_LIST:
        return output_create_list(output->value_nested, p_out);
    case CPYOUT_TYPE_TUPLE:
        return output_create_tuple(output->value_nested, p_out);
    }
    if (!out)
        return CPYOUT_FAILED_CONSTRUCTION;

    *p_out = out;
    return CPYOUT_SUCCESS;
}

static cpyutl_output_status_t output_create_tuple(const cpyutl_output_t outputs[], PyObject **const out)
{
    unsigned n = 0;
    while (outputs[n].type != CPYOUT_TYPE_NONE)
    {
        n += 1;
    }
    PyObject *const tuple = PyTuple_New(n);
    if (!tuple)
        return CPYOUT_FAILED_CONSTRUCTION;
    for (unsigned i = 0; i < n; ++i)
    {
        PyObject *val = NULL;
        const cpyutl_output_status_t res = output_create_value(outputs + i, &val);
        if (res != CPYOUT_SUCCESS)
        {
            Py_DECREF(tuple);
            return res;
        }
        PyTuple_SET_ITEM(tuple, i, val);
    }
    *out = tuple;
    return CPYOUT_SUCCESS;
}

static cpyutl_output_status_t output_create_list(const cpyutl_output_t outputs[], PyObject **const out)
{
    unsigned n = 0;
    while (outputs[n].type != CPYOUT_TYPE_NONE)
    {
        n += 1;
    }
    PyObject *const list = PyList_New(n);
    if (!list)
        return CPYOUT_FAILED_CONSTRUCTION;
    for (unsigned i = 0; i < n; ++i)
    {
        PyObject *val = NULL;
        const cpyutl_output_status_t res = output_create_value(outputs + i, &val);
        if (res != CPYOUT_SUCCESS)
        {
            Py_DECREF(list);
            return res;
        }
        PyList_SET_ITEM(list, i, val);
    }
    *out = list;
    return CPYOUT_SUCCESS;
}

static cpyutl_output_status_t output_create_dict(const cpyutl_output_t outputs[], PyObject **const out)
{
    unsigned n = 0;
    while (outputs[n].type != CPYOUT_TYPE_NONE)
    {
        n += 1;
    }
    PyObject *const dict = PyDict_New();
    if (!dict)
        return CPYOUT_FAILED_CONSTRUCTION;
    for (unsigned i = 0; i < n; ++i)
    {
        PyObject *val = NULL;
        const cpyutl_output_status_t res = output_create_value(outputs + i, &val);
        if (res != CPYOUT_SUCCESS)
        {
            Py_DECREF(dict);
            return res;
        }
        if (PyDict_SetItemString(dict, outputs[i].name, val) < 0)
        {
            Py_DECREF(dict);
            return CPYOUT_FAILED_DICT_INSERTION;
        }
    }
    *out = dict;
    return CPYOUT_SUCCESS;
}

static cpyutl_output_status_t validate_cpyutl_output_specs(const cpyutl_output_t outputs[], const int require_names)
{
    for (unsigned i = 0; outputs[i].type != CPYOUT_TYPE_NONE; ++i)
    {
        // Nested sequences require non-null contents
        if (outputs[i].type == CPYOUT_TYPE_DICT || outputs[i].type == CPYOUT_TYPE_LIST ||
            outputs[i].type == CPYOUT_TYPE_TUPLE)
        {
            if (outputs[i].value_nested == NULL)
            {
                fprintf(stderr, "Nested group of argument %u had no value.\n", i);
                return CPYOUT_NO_NESTED;
            }

            const cpyutl_output_status_t res =
                validate_cpyutl_output_specs(outputs[i].value_nested, outputs[i].type == CPYOUT_TYPE_DICT);
            if (res != CPYOUT_SUCCESS)
            {
                fprintf(stderr, "Failed nested group %u construction: %s.\n", i, cpyutl_output_status_str(res));
                return res;
            }
        }
        if (require_names && outputs[i].name == NULL)
        {
            fprintf(stderr, "Output %u had no name but it was required.\n", i);
            return CPYOUT_INVALID_NO_NAME;
        }
        if (outputs[i].type == CPYOUT_TYPE_PYOBJ && outputs[i].value_obj == NULL)
        {
            fprintf(stderr, "Output %u had no value but it was a PyObject.\n", i);
            return CPYOUT_INVALID_VALUE;
        }
        if (outputs[i].type == CPYOUT_TYPE_PYSTRING && outputs[i].value_str == NULL)
        {
            fprintf(stderr, "Output %u had no value but it was a string.\n", i);
            return CPYOUT_INVALID_VALUE;
        }
    }

    return CPYOUT_SUCCESS;
}

cpyutl_output_status_t cpyutl_output_create(const cpyutl_output_type_t out_type, const cpyutl_output_t outputs[],
                                            PyObject **const out)
{
    // Check we have outputs
    CPYUTL_ASSERT(outputs != NULL, "Output specification must not be null.");
    // Check the output type is correct
    CPYUTL_ASSERT((out_type == CPYOUT_TYPE_DICT) || (out_type == CPYOUT_TYPE_LIST) || (out_type == CPYOUT_TYPE_TUPLE),
                  "Output type must be either PyDictObject, PyListObject or PyTupleObject (%s was given).",
                  cpyutl_output_type_str(out_type));
    CPYUTL_ASSERT(validate_cpyutl_output_specs(outputs, out_type == CPYOUT_TYPE_DICT) == CPYOUT_SUCCESS,
                  "Output specifications were not valid (author is retarded).");
    (void)validate_cpyutl_output_specs;

    switch (out_type)
    {
    case CPYOUT_TYPE_DICT:
        return output_create_dict(outputs, out);
    case CPYOUT_TYPE_LIST:
        return output_create_list(outputs, out);
    case CPYOUT_TYPE_TUPLE:
        return output_create_tuple(outputs, out);
    default:
        return CPYOUT_INVALID_OUT_TYPE;
    }
}

int cpyutl_traverse_heap_type(PyObject *op, const visitproc visit, void *arg)
{
    Py_VISIT(Py_TYPE(op));
    return 0;
}

static const char *arg_status_strings[] = {
    [CPYARG_SUCCESS] = "Parsed correctly",
    [CPYARG_MISSING] = "Argument was missing",
    [CPYARG_INVALID] = "Argument had invalid value",
    [CPYARG_DUPLICATE] = "Argument was found twice",
    [CPYARG_BAD_SPECS] = "Specifications were incorrect",
    [CPYARG_KW_AS_POS] = "Keyword argument was specified as a positional argument",
    [CPYARG_NO_KW] = "No argument has this keyword",
    [CPYARG_UNKNOWN] = "Unknown error",
    [CPYARG_KW_IN_SEQUENCE] = "Keyword argument was found in a sequence",
};

const char *cpyutl_argument_status_str(const cpyutl_argument_status_t e)
{
    if ((size_t)e >= (sizeof(arg_status_strings) / sizeof(arg_status_strings[0])))
        return "UNKNOWN";
    return arg_status_strings[e];
}

void cpyutl_failure_exit(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    exit(1);
}

PyTypeObject *cpyutl_add_type_from_spec_to_module(PyObject *module, PyType_Spec *spec, PyObject *bases)
{
    PyTypeObject *const type = (PyTypeObject *)PyType_FromMetaclass(NULL, module, spec, bases);
    if (!type)
    {
        return NULL;
    }
    const char *name = spec->name;

    const char *last_pos = strrchr(name, '.');
    if (last_pos)
        name = last_pos + 1;

    const int res = PyModule_AddObjectRef(module, name, (PyObject *)type);
    Py_DECREF(type);

    if (res < 0)
    {
        return NULL;
    }

    return type;
}
