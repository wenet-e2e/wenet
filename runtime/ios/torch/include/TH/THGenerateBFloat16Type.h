#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateBFloat16Type.h"
#endif

#include <c10/util/BFloat16.h>
#define scalar_t at::BFloat16
#define accreal double
#define Real BFloat16
#define TH_REAL_IS_BFLOAT16
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef accreal
#undef scalar_t
#undef Real
#undef TH_REAL_IS_BFLOAT16

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
