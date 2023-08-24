CUDA_PATH       = /usr/local/cuda
TRT_PATH        = /usr/lib/x86_64-linux-gnu
NVCC            = $(CUDA_PATH)/bin/nvcc
#SM              = 61
                # 61 for GTX1070, 75 for T4,80 for A30
GENCODE         = -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86
CUFLAG          = -w -std=c++14 -O3 -UDEBUG -Xcompiler -fPIC $(GENCODE)
CPPFLAG         = -w -std=c++14 -O3 -use_fast_math
SOFLAG          = $(CUFLAG) -shared
INCLUDE         = -I. -I$(CUDA_PATH)/include
LDFLAG          = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcublasLt -L$(TRT_PATH)/lib -lnvinfer

SRC_CU          = $(shell find ./ -name '*.cu')

all: LayerNorm.so

%.o: %.cu
	$(NVCC) $(CUFLAG) $(INCLUDE) -o $@ -c $<

LayerNorm.so: $(SRC_CU:.cu=.o)
	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $^

.PHONY: clean
clean:
	rm -rf ./*.so ./*.o ./*.d ./*.trt

.PHONY: test
test:
	clear
	python testLayerNormPlugin.py

