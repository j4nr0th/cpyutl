

list(APPEND
        MODULE_SOURCES
        module.c
)

Python_add_library(_cpyutl_test
        MODULE
        ${MODULE_SOURCES}
        WITH_SOABI
)
install(TARGETS _cpyutl_test DESTINATION ${SKBUILD_PROJECT_NAME})
target_link_libraries(_cpyutl_test PRIVATE cpyutl)
if ("${NumPy_INCLUDE_DIRS}" STREQUAL "")
    message(FATAL_ERROR "NumPy_INCLUDE_DIRS was empty.")
endif ()
target_include_directories(_cpyutl_test PRIVATE ${NumPy_INCLUDE_DIRS} ${Python_INCLUDE_DIRS})
