#include <torch/csrc/cuda/undef_macros.h>

#define THWStoragePtr THCStoragePtr
#define THPStoragePtr THCPStoragePtr
#define THWTensorPtr THCTensorPtr
#define THPTensorPtr THCPTensorPtr

#define THWStorage THCStorage
#define THWStorage_(NAME) THCStorage_(NAME)
#define THWTensor THCTensor
#define THWTensor_(NAME) THCTensor_(NAME)

#define THPStorage_(NAME) TH_CONCAT_4(THCP,Real,Storage_,NAME)
#define THPStorageBaseStr THCPStorageBaseStr
#define THPStorageStr THCPStorageStr
#define THPStorageClass THCPStorageClass
#define THPStorageType THCPStorageType

#define THPTensor_(NAME) TH_CONCAT_4(THCP,Real,Tensor_,NAME)
#define THPTensor_stateless_(NAME) TH_CONCAT_4(THCP,Real,Tensor_stateless_,NAME)
#define THPTensor THCPTensor
#define THPTensorStr THCPTensorStr
#define THPTensorBaseStr THCPTensorBaseStr
#define THPTensorClass THCPTensorClass
#define THPTensorType THCPTensorType

#define THPTensorStatelessType THCPTensorStatelessType
#define THPTensorStateless THCPTensorStateless


#define THSPTensorPtr THCSPTensorPtr

#define THSPTensor_(NAME) TH_CONCAT_4(THCSP,Real,Tensor_,NAME)
#define THSPTensor_stateless_(NAME) TH_CONCAT_4(THCSP,Real,Tensor_stateless_,NAME)
#define THSPTensor THCSPTensor
#define THSPTensorStr THCSPTensorStr
#define THSPTensorBaseStr THCSPTensorBaseStr
#define THSPTensorClass THCSPTensorClass
#define THSPTensorType THCSPTensorType

#define THSPTensorStatelessType THCSPTensorStatelessType
#define THSPTensorStateless THCSPTensorStateless


#define LIBRARY_STATE_NOARGS state
#define LIBRARY_STATE state,
#define LIBRARY_STATE_TYPE THCState*,
#define LIBRARY_STATE_TYPE_NOARGS THCState*
#define TH_GENERIC_FILE THC_GENERIC_FILE

#define THHostTensor TH_CONCAT_3(TH,Real,Tensor)
#define THHostTensor_(NAME) TH_CONCAT_4(TH,Real,Tensor_,NAME)
#define THHostStorage TH_CONCAT_3(TH,Real,Storage)
#define THHostStorage_(NAME) TH_CONCAT_4(TH,Real,Storage_,NAME)
