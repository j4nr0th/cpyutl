#ifndef CPYUTL_LIBRARY_H
#define CPYUTL_LIBRARY_H

#ifdef __GNUC__
// Import/export
#define CPYUTL_INTERNAL __attribute__((visibility("hidden")))
#define CPYUTL_EXPORT __attribute__((visibility("default")))
// Noreturn
#define CPYUTL_NORETURN __attribute__((noreturn))
#endif

#ifndef CPYUTL_INTERNAL
#define CPYUTL_INTERNAL
#endif

#include <Python.h>

#ifdef PY_ARRAY_UNIQUE_SYMBOL
/**
 * @brief Validates a NumPy array based on the specified dimensions, data type, and flags.
 *
 * This function checks several conditions for the given array, including
 * - Whether the array has the required flags.
 * - Whether the number of dimensions matches the expected value.
 * - Whether the data type matches the expected type (if specified).
 * - Whether each dimension size matches the expected size (if specified).
 *
 * If any of the conditions fail, a Python exception is raised with a descriptive error message,
 * and the function returns -1. Otherwise, the function returns 0 on success.
 *
 * @param arr Pointer to the NumPy array object to be validated.
 * @param n_dim The expected number of dimensions for the array.
 * @param dims Array of expected sizes for each dimension. Use 0 for dimensions that do not require strict matching.
 * @param dtype The expected data type of the array (e.g., NPY_DOUBLE). Use a negative value to skip this check.
 * @param flags The required flags that must be present in the array (e.g., NPY_ARRAY_C_CONTIGUOUS).
 * @param name The name of the array (for error messages).
 * @return Returns 0 if the array passes all validation checks, or -1 if any check fails.
 *         In the event of failure, a Python exception is set with an appropriate error message.
 */
#ifdef __GNUC__
__attribute__((unused))
#endif
static int
check_input_array(const PyArrayObject *const arr, const unsigned n_dim, const npy_intp dims[static n_dim],
                  const int dtype, const int flags, const char *name)
{
    if (!PyArray_Check(arr))
    {
        PyErr_Format(PyExc_TypeError, "Object %s is not a numpy array, but is %R.", name, Py_TYPE(arr));
        return -1;
    }

    if (name == NULL)
    {
        name = "UNKNOWN";
    }

    const int arr_flags = PyArray_FLAGS(arr);
    if ((arr_flags & flags) != flags)
    {
        PyErr_Format(PyExc_ValueError, "Array %s flags %u don't contain required flags %u.", name, arr_flags, flags);
        return -1;
    }

    if (n_dim && PyArray_NDIM(arr) != (npy_intp)n_dim)
    {
        PyErr_Format(PyExc_ValueError,
                     "Number of dimensions for the array %s does not match expected value (expected %u, got %u).", name,
                     n_dim, (unsigned)PyArray_NDIM(arr));
        return -1;
    }

    if (dtype >= 0 && PyArray_TYPE(arr) != dtype)
    {
        PyErr_Format(PyExc_ValueError, "Array %s does not have the expected type (expected %u, got %u).", name, dtype,
                     PyArray_TYPE(arr));
        return -1;
    }

    const npy_intp *const d = PyArray_DIMS(arr);
    for (unsigned i_dim = 0; i_dim < n_dim; ++i_dim)
    {
        if (dims[i_dim] != 0 && d[i_dim] != dims[i_dim])
        {
            PyErr_Format(PyExc_ValueError,
                         "Array %s dimension %u of the did not match expected value (expected %u, got %u).", name,
                         i_dim, dims[i_dim], d[i_dim]);
            return -1;
        }
    }

    return 0;
}
#endif

/**
 * @brief Raises a new Python exception, preserving the current exception context.
 *
 * This function raises a new Python exception with the specified type while
 * preserving the context of the current exception if one exists. It formats
 * the error message using the provided format string and additional arguments.
 * If an exception is already set, it will be attached as the cause of the new
 * exception. If no exception is set, a new exception is generated with the
 * specified type and formatted message.
 *
 * @param exception[in] The Python exception type to raise (e.g., PyExc_RuntimeError).
 * @param format[in] The printf-style format string to create the exception message.
 * @param ...[in] Additional arguments to populate the format string.
 */
[[gnu::format(printf, 2, 3)]]
CPYUTL_INTERNAL void raise_exception_from_current(PyObject *exception, const char *format, ...);

typedef enum
{
    CPYARG_TYPE_NONE,     // Terminator of the list
    CPYARG_TYPE_SSIZE,    // integer represented as Py_ssize_t
    CPYARG_TYPE_BOOL,     // true/false represented as int
    CPYARG_TYPE_DOUBLE,   // floating point number as double
    CPYARG_TYPE_STRING,   // null-terminated string of characters as const char*
    CPYARG_TYPE_PYTHON,   // PyObject, which may be type-checked
    CPYARG_TYPE_CUSTOM,   // Call a custom function which handles conversion
    CPYARG_TYPE_SEQUENCE, // Contains a sequence of arguments with no keywords
} cpyutl_argument_type_t;

typedef enum
{
    CPYARG_SUCCESS,        // Parsed correctly
    CPYARG_MISSING,        // Argument was missing
    CPYARG_INVALID,        // Argument had invalid value
    CPYARG_DUPLICATE,      // Argument was found twice
    CPYARG_BAD_SPECS,      // Specifications were incorrect
    CPYARG_KW_AS_POS,      // Keyword argument was specified as a positional argument
    CPYARG_NO_KW,          // No argument has this keyword
    CPYARG_UNKNOWN,        // Unknown error
    CPYARG_KW_IN_SEQUENCE, // Keyword argument was found in a sequence
} cpyutl_argument_status_t;

typedef struct
{
    cpyutl_argument_type_t type; // Type of the argument
    void *p_val;                 // Pointer used to return the value (or for the callback)
    const char *kwname;          // Keyword name or NULL for positional-only
    int optional;                // Non-zero for optional arguments
    int kw_only;                 // Non-zero for keyword-only
    union {
        PyTypeObject *type_check;                                            // For type=ARG_TYPE_PYTHON
        int (*custom_convert)(PyObject *obj, void *p_val, const char *name); // For type=ARG_TYPE_CUSTOM
    };
    int *p_found; // Used to track if non-optional arguments were found.
} cpyutl_argument_t;

CPYUTL_INTERNAL
const char *cpyutl_argument_status_str(cpyutl_argument_status_t e);

/**
 * @brief Parses positional and keyword arguments and assigns values to the specified argument descriptors.
 *
 * This function processes the provided arguments and matches them with the specified
 * argument descriptions in the `specs` array. It ensures that the arguments conform
 * to the specifications, validates them, and assigns their values to the associated
 * fields in the `specs` array. Positional and keyword arguments are handled separately
 * while checking for duplicates, missing arguments, and type mismatches.
 *
 * @param specs[in,out] An array of argument_t structures that specify the required
 * and optional arguments. This array will be updated to reflect the presence of parsed arguments.
 * @param args[in] An array of Python objects representing the positional arguments passed
 * to the function. Must not be NULL.
 * @param nargs[in] The total number of positional arguments provided in `args`. Must not be negative.
 * @param kwnames[in] A tuple of Python strings representing the names of the keyword arguments.
 * If no keyword arguments exist, this may be NULL.
 *
 * @return An argument_status_t value indicating the result of parsing the arguments:
 *         - ARG_STATUS_SUCCESS: If all arguments were parsed successfully.
 *         - ARG_STATUS_MISSING: If a required argument was not provided.
 *         - ARG_STATUS_INVALID: If an argument failed type or value validation.
 *         - ARG_STATUS_DUPLICATE: If an argument was passed multiple times.
 *         - ARG_STATUS_BAD_SPECS: If the argument specifications were invalid.
 *         - ARG_STATUS_KW_AS_POS: If a keyword-only argument was provided as positional.
 *         - ARG_STATUS_NO_KW: If a keyword argument did not match any specified arguments.
 *         - ARG_STATUS_UNKNOWN: If an unknown error occurred.
 *         - ARG_STATUS_KW_IN_SEQUENCE: If a keyword argument was provided within a sequence.
 */
CPYUTL_INTERNAL
cpyutl_argument_status_t parse_arguments(cpyutl_argument_t specs[], PyObject *const args[], Py_ssize_t nargs,
                                         const PyObject *kwnames);

/**
 * @brief Wrapper around `parse_arguments`, which raises a Python exception if parsing fails.
 *
 * @param specs[in,out] An array of argument_t structures that specify the required
 * and optional arguments. This array will be updated to reflect the presence of parsed arguments.
 * @param args[in] An array of Python objects representing the positional arguments passed
 * to the function. Must not be NULL.
 * @param nargs[in] The total number of positional arguments provided in `args`. Must not be negative.
 * @param kwnames[in] A tuple of Python strings representing the names of the keyword arguments.
 * If no keyword arguments exist, this may be NULL.
 *
 * @return Returns 0 if the arguments are valid and properly parsed; otherwise, returns -1 and
 *         raises an appropriate Python exception to indicate the specific error.
 */
static inline int parse_arguments_check(cpyutl_argument_t specs[], PyObject *const args[], const Py_ssize_t nargs,
                                        const PyObject *kwnames)
{
    const cpyutl_argument_status_t res = parse_arguments(specs, args, nargs, kwnames);
    if (res != CPYARG_SUCCESS)
    {
        raise_exception_from_current(PyExc_TypeError, "Invalid arguments to function (%s).",
                                     cpyutl_argument_status_str(res));
        return -1;
    }
    return 0;
}

typedef enum
{
    CPYOUT_TYPE_NONE,     // Terminate the list of output values
    CPYOUT_TYPE_PYINT,    // Python's int
    CPYOUT_TYPE_PYFLOAT,  // Python's float
    CPYOUT_TYPE_PYSTRING, // Python's string
    CPYOUT_TYPE_PYBOOL,   // Python's bool
    CPYOUT_TYPE_PYOBJ,    // PyObject (*do not* steal the reference)
    CPYOUT_TYPE_LIST,     // Nested values go to a list
    CPYOUT_TYPE_TUPLE,    // Nested values go to a tuple
    CPYOUT_TYPE_DICT,     // Nested values go to a dict
} cpyutl_output_type_t;
CPYUTL_INTERNAL
const char *cpyutl_output_type_str(cpyutl_output_type_t e);

typedef enum
{
    CPYOUT_SUCCESS,
    CPYOUT_FAILED_CONSTRUCTION,
    CPYOUT_INVALID_SPEC,
    CPYOUT_FAILED_DICT_INSERTION,
    CPYOUT_INVALID_OUT_TYPE,
    CPYOUT_INVALID_NO_NAME,
    CPYOUT_NO_NESTED,
    CPYOUT_INVALID_VALUE,
} cpyutl_output_status_t;

CPYUTL_INTERNAL
const char *cpyutl_output_status_str(cpyutl_output_status_t e);

typedef struct cpyutl_output_t cpyutl_output_t;
struct cpyutl_output_t
{
    cpyutl_output_type_t type; // Type of the value
    union {
        Py_ssize_t value_int;          // Used by TYPE_PYINT
        double value_float;            // Used by TYPE_PYFLOAT
        const char *value_str;         // Used by TYPE_PYSTRING
        int value_bool;                // Used by TYPE_BOOL
        PyObject *value_obj;           // Used by TYPE_PYOBJ
        cpyutl_output_t *value_nested; // Used by TYPE_LIST, TYPE_TUPLE, and TYPE_DICT
    };
    const char *name; // Must be non-NULL when part of a dict.
};

/**
 * @brief Creates a Python object (tuple, list, or dictionary) based on the specified type and output specifications.
 *
 * This function constructs a Python object of the specified type (e.g., tuple, list, or dictionary)
 * and populates it with values as described in the output specifications. If the output type is not
 * valid or the output specifications are incorrect, the function returns an appropriate error code.
 *
 * @param out_type[in] The type of Python object to create (e.g., CPYOUT_TYPE_TUPLE, CPYOUT_TYPE_LIST,
 * CPYOUT_TYPE_DICT).
 * @param outputs[in] Array of output specifications, defining the values to include in the resulting Python object.
 *                    Should not be null and must comply with the requirements for the specified `out_type`.
 * @param out[out] Pointer to a PyObject pointer where the created Python object will be stored. Must not be null.
 *
 * @return cpyutl_output_status_t Status code indicating success (CPYOUT_SUCCESS) or the specific error
 *                                encountered during object creation (e.g., CPYOUT_INVALID_SPEC,
 * CPYOUT_FAILED_CONSTRUCTION).
 */
CPYUTL_INTERNAL
cpyutl_output_status_t cpyutl_output_create(cpyutl_output_type_t out_type, const cpyutl_output_t outputs[],
                                            PyObject **out);

/**
 * @brief Wrapper around `cpyutl_output_create`, which raises a Python exception if parsing fails.
 *
 * @param out_type[in] The type of Python object to create (e.g., CPYOUT_TYPE_TUPLE, CPYOUT_TYPE_LIST,
 * CPYOUT_TYPE_DICT).
 * @param outputs[in] Array of output specifications, defining the values to include in the resulting Python object.
 *                    Should not be null and must comply with the requirements for the specified `out_type`.
 *
 * @return Non-NULL when `cpyutl_output_create` succeeds, otherwise NULL with a raised Python exception.
 */
static inline PyObject *cpyutl_output_create_check(const cpyutl_output_type_t out_type, const cpyutl_output_t outputs[])
{
    PyObject *out;
    const cpyutl_output_status_t res = cpyutl_output_create(out_type, outputs, &out);
    if (res != CPYOUT_SUCCESS)
    {
        raise_exception_from_current(PyExc_TypeError, "Failed to create output tuple.");
        return NULL;
    }
    return out;
}

/**
 * @brief Function that can be used by heap types as tp_traverse slot if no special behavior is needed.
 */
CPYUTL_INTERNAL
int cpyutl_traverse_heap_type(PyObject *op, visitproc visit, void *arg);

#ifdef __GNUC__
__attribute__((format(printf, 1, 2)))
#endif
CPYUTL_INTERNAL CPYUTL_NORETURN void
cpyutl_failure_exit(const char *fmt, ...);

#ifndef CPYUTL_ENABLE_ASSERTS
#define CPYUTL_ASSERT(cond, fmt, ...) (void)0
#else

#define CPYUTL_ASSERT(cond, fmt, ...)                                                                                  \
    ((cond) ? (void)0                                                                                                  \
            : cpyutl_failure_exit("%s:%d (%s): Failed assertion \"%s\": " fmt "\n", __FILE__, __LINE__, __func__,      \
                                  #cond __VA_OPT__(, )##__VA_ARGS__))

#endif

/**
 * @brief Adds a new type to a Python module from a given type specification.
 *
 * This function creates a Python type object from the provided type
 * specification and optionally, a base type or tuple of base types. The
 * resulting type is added to the specified module with a name derived
 * from the given type specification. If the addition to the module fails,
 * the function returns NULL.
 *
 * @param module[in] The target Python module to which the type will be added.
 * @param spec[in] The type specification describing the new type to create.
 * @param bases[in] Optional base type or tuple of base types for the new type,
 *                  or NULL for default base types.
 *
 * @return A pointer to the newly created Python type object, or NULL if the
 *         type creation or module addition fails.
 */
CPYUTL_INTERNAL
PyTypeObject *cpyutl_add_type_from_spec_to_module(PyObject *module, PyType_Spec *spec, PyObject *bases);

#endif // CPYUTL_LIBRARY_H
