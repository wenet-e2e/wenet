// Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef RUNTIME_GPU_TENSORRT_LAYERNORMPLUGIN_LAYERNORMPLUGIN_H_
#define RUNTIME_GPU_TENSORRT_LAYERNORMPLUGIN_LAYERNORMPLUGIN_H_

#include <vector>
#include <string>
#include <cassert>
#include <NvInfer.h>
#include <cuda_fp16.h> // NOLINT
#include <cub/cub.cuh> // NOLINT

#define CEIL_DIVIDE(X, Y)    (((X)+(Y)-1)/(Y))
#define CEIL_TO(X, Y)        (((X)+(Y)-1)/(Y)*(Y))

template <typename T>
__device__ T epsilon();

template <>
__device__ float epsilon<float>() {
    return (float)6.0e-12; // NOLINT
}

template <>
__device__ half epsilon<half>() {
    return (half)6.0e-6;
}

// +------- Debug wrapper -----------------------------------
#if DEBUG
#define WHERE_AM_I() do {printf("[%s]:this=->%p\n", __func__, this);} while (0);
#else
#define WHERE_AM_I()
#endif  // DEBUG

// +------- Plguin -------------------------------------------
namespace { // NOLINT
static const char* PLUGIN_NAME{"LayerNorm"};
static const char* PLUGIN_VERSION{"1"};
}  // namespace

namespace nvinfer1 {

// +------- Plugin body ---------------------------------------
class LayerNormPlugin: public IPluginV2DynamicExt {
 private:
    std::string name_;
    std::string namespace_;

 public:
    LayerNormPlugin(const std::string& name) : name_(name) { // NOLINT
        WHERE_AM_I();
    }

    LayerNormPlugin(const std::string& name,
                    const void* data, size_t length) : name_(name) {
        WHERE_AM_I();
    }

    LayerNormPlugin() = delete;

    ~LayerNormPlugin() {
        WHERE_AM_I();
    }

    size_t getSerializationSize() const noexcept override {
        WHERE_AM_I();
        return 0;
    }

    void serialize(void *buffer) const noexcept override {
        WHERE_AM_I();
    }

    IPluginV2DynamicExt* clone() const noexcept override {
        WHERE_AM_I();
        return new LayerNormPlugin(name_);
    }

    int getNbOutputs() const noexcept override {
        WHERE_AM_I();
        return 1;
    }

    DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs,
                                  int32_t nbInputs,
                                  IExprBuilder& exprBuilder) noexcept override {
        WHERE_AM_I();
        return inputs[0];
    }

    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut,
                                   int32_t nbInputs,
                                   int32_t nbOutputs) noexcept override {
        WHERE_AM_I();
        if (inOut[pos].format != TensorFormat::kLINEAR) {
            return false;
        }

        bool res = false;
        switch (pos) {
        case 0:
            res = (inOut[pos].type == DataType::kFLOAT
                   || inOut[pos].type == DataType::kHALF); break;
        case 1:
        case 2:
        case 3:
            res = inOut[pos].type == inOut[0].type; break;
        default:  // should NOT be here
            res = false; break;
        }

        return res;
    }

    DataType getOutputDataType(int outputIndex,
                               const DataType* inputTypes,
                               int nbInputs) const noexcept override {
        WHERE_AM_I();
        return inputTypes[0];
    }

    void configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,
                         const DynamicPluginTensorDesc* out,
                         int32_t nbOutputs) noexcept override {
        WHERE_AM_I();
    }

    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs,
                            const PluginTensorDesc* outputs,
                            int32_t nbOutputs) const noexcept override {
        WHERE_AM_I();
        return 0;
    }

    void setPluginNamespace(const char* szNamespace) noexcept override {
        WHERE_AM_I();
        namespace_ = szNamespace;
    }
    const char* getPluginNamespace() const noexcept override {
        WHERE_AM_I();
        return namespace_.c_str();
    }
    const char* getPluginType() const noexcept override {
        WHERE_AM_I();
        return PLUGIN_NAME;
    }
    const char* getPluginVersion() const noexcept override {
        WHERE_AM_I();
        return PLUGIN_VERSION;
    }
    int initialize() noexcept override {
        WHERE_AM_I();
        return 0;
    }
    void terminate() noexcept override {
        WHERE_AM_I();
        return;
    }

    void destroy() noexcept override {
        WHERE_AM_I();
    }

    int32_t enqueue(const PluginTensorDesc* inputDesc,
                    const PluginTensorDesc* outputDesc,
                    const void* const* inputs,
                    void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override;
};  // class LayerNormPlugin

class LayerNormPluginCreator : public IPluginCreator {
 private:
    static PluginFieldCollection fc_;
    static std::vector<PluginField> attr_;
    std::string namespace_;

 public:
    LayerNormPluginCreator() {
        fc_.nbFields = attr_.size();
        fc_.fields = attr_.data();
    }

    ~LayerNormPluginCreator() {}

    IPluginV2* createPlugin(const char* name,
                            const PluginFieldCollection* fc) noexcept override {
        WHERE_AM_I();
        return new LayerNormPlugin(name);
    }

    IPluginV2* deserializePlugin(const char* name, const void* serialData,
                                 size_t serialLength) noexcept override {
        return new LayerNormPlugin(name, serialData, serialLength);
    }

    void setPluginNamespace(const char* szNamespace) noexcept override {
        namespace_ = szNamespace;
    }

    const char* getPluginNamespace() const noexcept override {
        return namespace_.c_str();
    }

    const char* getPluginName() const noexcept override {
        return PLUGIN_NAME;
    }

    const char* getPluginVersion() const noexcept override {
        return PLUGIN_VERSION;
    }

    const PluginFieldCollection* getFieldNames() noexcept override {
        return &fc_;
    }
};  // class LayerNormPluginCreator

}  // namespace nvinfer1
#endif  // RUNTIME_GPU_TENSORRT_LAYERNORMPLUGIN_LAYERNORMPLUGIN_H_
