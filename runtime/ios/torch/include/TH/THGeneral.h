#ifndef TH_GENERAL_INC
#define TH_GENERAL_INC

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <time.h>
#include <string.h>
#include <stddef.h>
#include <inttypes.h>

#ifdef TH_BLAS_MKL
#include <mkl_vsl.h>
#endif

#define USE_BLAS
/* #undef USE_LAPACK */
/* #undef BLAS_F2C */
/* #undef BLAS_USE_CBLAS_DOT */

# define TH_EXTERNC extern "C"

// Note(jiayq): copied from ATen/core/Macros.h. Because internal build of TH
// and ATen are not unified yet, we need to duplicate code for now. Long term
// we should merge macros.
#ifdef _WIN32
#if !defined(AT_CORE_STATIC_WINDOWS)
// TODO: unify the controlling macros.
#if defined(CAFFE2_BUILD_MAIN_LIBS) || defined(ATen_cpu_EXPORTS) || defined(caffe2_EXPORTS)
#define TH_CPP_API __declspec(dllexport)
#else // defined(CAFFE2_BUILD_MAIN_LIBS) || defined(ATen_cpu_EXPORTS) || defined(caffe2_EXPORTS)
#define TH_CPP_API __declspec(dllimport)
#endif // defined(CAFFE2_BUILD_MAIN_LIBS) || defined(ATen_cpu_EXPORTS) || defined(caffe2_EXPORTS)
#else // !defined(AT_CORE_STATIC_WINDOWS)
#define TH_CPP_API
#endif // !defined(AT_CORE_STATIC_WINDOWS)
#else  // _WIN32
#if defined(__GNUC__)
#define TH_CPP_API __attribute__((__visibility__("default")))
#endif // defined(__GNUC__)
#endif  // _WIN32

#ifdef NO_EXPORT
#undef TH_CPP_API
#define TH_CPP_API
#endif

#define TH_API TH_CPP_API

#ifdef _WIN32
# define TH_NO_RETURN __declspec(noreturn)
# define TH_UNREACHABLE
#else
# define TH_NO_RETURN __attribute__((noreturn))
# define TH_UNREACHABLE __builtin_unreachable();
#endif

#if defined(__GNUC__) && ((__GNUC__ > 2) || (__GNUC__ == 2 && __GNUC_MINOR__ > 4))
# define TH_UNUSED __attribute__((unused))
#else
# define TH_UNUSED
#endif

typedef void (*THErrorHandlerFunction)(const char *msg, void *data);
typedef void (*THArgErrorHandlerFunction)(int argNumber, const char *msg, void *data);

#define TH_DESC_BUFF_LEN 64
typedef struct {
    char str[TH_DESC_BUFF_LEN];
} THDescBuff;


TH_API THDescBuff _THSizeDesc(const int64_t *size, const int64_t ndim);
TH_API TH_NO_RETURN void _THError(const char *file, const int line, const char *fmt, ...);
TH_API void _THAssertionFailed(const char *file, const int line, const char *exp, const char *fmt, ...);
TH_API void THSetErrorHandler(THErrorHandlerFunction new_handler, void *data);
TH_API void THSetDefaultErrorHandler(THErrorHandlerFunction new_handler, void *data);
TH_API void _THArgCheck(const char *file, int line, int condition, int argNumber, const char *fmt, ...);
TH_API void THSetArgErrorHandler(THArgErrorHandlerFunction new_handler, void *data);
TH_API void THSetDefaultArgErrorHandler(THArgErrorHandlerFunction new_handler, void *data);
TH_API void* THAlloc(ptrdiff_t size);
TH_API void* THRealloc(void *ptr, ptrdiff_t size);
TH_API void THFree(void *ptr);
TH_API void THSetGCHandler( void (*torchGCHandlerFunction)(void *data), void *data );
// this hook should only be called by custom allocator functions
TH_API void THHeapUpdate(ptrdiff_t size);

#define THError(...) _THError(__FILE__, __LINE__, __VA_ARGS__)

#define THCleanup(...) __VA_ARGS__

#define THArgCheck(...)                                               \
do {                                                                  \
  _THArgCheck(__FILE__, __LINE__, __VA_ARGS__);                       \
} while(0)

#define THArgError(...)                                               \
do {                                                                  \
  _THArgCheck(__FILE__, __LINE__, false, __VA_ARGS__);                \
  TH_UNREACHABLE                                                      \
} while(0)

#define THArgCheckWithCleanup(condition, cleanup, ...)                \
do if (!(condition)) {                                                \
  cleanup                                                             \
  _THArgCheck(__FILE__, __LINE__, 0, __VA_ARGS__);                    \
} while(0)

#define THAssert(exp)                                                 \
do {                                                                  \
  if (!(exp)) {                                                       \
    _THAssertionFailed(__FILE__, __LINE__, #exp, "");                 \
  }                                                                   \
} while(0)

#define THAssertMsg(exp, ...)                                         \
do {                                                                  \
  if (!(exp)) {                                                       \
    _THAssertionFailed(__FILE__, __LINE__, #exp, __VA_ARGS__);        \
  }                                                                   \
} while(0)

#define TH_CONCAT_STRING_2(x,y) TH_CONCAT_STRING_2_EXPAND(x,y)
#define TH_CONCAT_STRING_2_EXPAND(x,y) #x #y

#define TH_CONCAT_STRING_3(x,y,z) TH_CONCAT_STRING_3_EXPAND(x,y,z)
#define TH_CONCAT_STRING_3_EXPAND(x,y,z) #x #y #z

#define TH_CONCAT_STRING_4(x,y,z,w) TH_CONCAT_STRING_4_EXPAND(x,y,z,w)
#define TH_CONCAT_STRING_4_EXPAND(x,y,z,w) #x #y #z #w

#define TH_CONCAT_2(x,y) TH_CONCAT_2_EXPAND(x,y)
#define TH_CONCAT_2_EXPAND(x,y) x ## y

#define TH_CONCAT_3(x,y,z) TH_CONCAT_3_EXPAND(x,y,z)
#define TH_CONCAT_3_EXPAND(x,y,z) x ## y ## z

#define TH_CONCAT_4_EXPAND(x,y,z,w) x ## y ## z ## w
#define TH_CONCAT_4(x,y,z,w) TH_CONCAT_4_EXPAND(x,y,z,w)

#define THMin(X, Y)  ((X) < (Y) ? (X) : (Y))
#define THMax(X, Y)  ((X) > (Y) ? (X) : (Y))

#if (defined(_MSC_VER) || defined(__MINGW32__))
#define snprintf _snprintf
#define popen _popen
#define pclose _pclose
#include <BaseTsd.h>
#if !defined(HAVE_SSIZE_T)
typedef SSIZE_T ssize_t;
#endif
#endif

#endif
