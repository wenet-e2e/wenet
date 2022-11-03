#pragma once

#include <ATen/core/ATenGeneral.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/core/ATenGeneral.h>
#include <ATen/core/Generator.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/detail/HIPHooksInterface.h>
#include <c10/util/Exception.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/QEngine.h>

#include <memory>
#include <mutex>
#include <cstdint>

namespace at {

class Tensor;

class TORCH_API Context {
 public:
  Context();

  const Generator& defaultGenerator(Device device) {
    DeviceType device_type = device.type();
    initCUDAIfNeeded(device_type);
    initHIPIfNeeded(device_type);
    if (device_type == at::kCPU) {
      return at::detail::getDefaultCPUGenerator();
    } else if (device_type == at::kCUDA) {
      return at::detail::getCUDAHooks().getDefaultCUDAGenerator(device.index());
    } else {
      AT_ERROR(DeviceTypeName(device_type), " device type not enabled.");
    }
  }
  Device getDeviceFromPtr(void* data, DeviceType device_type) {
    initCUDAIfNeeded(device_type);
    initHIPIfNeeded(device_type);
    if (device_type == at::kCPU) {
      return DeviceType::CPU;
    } else if (device_type == at::kCUDA) {
      return at::detail::getCUDAHooks().getDeviceFromPtr(data);
    } else {
      AT_ERROR(DeviceTypeName(device_type), " device type not enabled.");
    }
  }
  static bool isPinnedPtr(void* data) {
    return detail::getCUDAHooks().isPinnedPtr(data);
  }
  static bool hasOpenMP() ;
  static bool hasMKL() ;
  static bool hasLAPACK() ;
  static bool hasMKLDNN() ;
  static bool hasMAGMA() {
    return detail::getCUDAHooks().hasMAGMA();
  }
  static bool hasCUDA() {
    return detail::getCUDAHooks().hasCUDA();
  }
  static bool hasCUDART() {
    return detail::getCUDAHooks().hasCUDART();
  }
  static long versionCUDART() {
    return detail::getCUDAHooks().versionCUDART();
  }
  static bool hasHIP() {
    return detail::getHIPHooks().hasHIP();
  }
  static bool hasXLA() {
    return c10::impl::hasDeviceGuardImpl(at::DeviceType::XLA);
  }
  static bool hasMLC() {
    return c10::impl::hasDeviceGuardImpl(at::DeviceType::MLC);
  }
  // defined in header so that getNonVariableType has ability to inline
  // call_once check. getNonVariableType is called fairly frequently
  THCState* lazyInitCUDA() {
    std::call_once(thc_init,[&] {
      thc_state = detail::getCUDAHooks().initCUDA();
    });
    return thc_state.get();
  }
  THHState* lazyInitHIP() {
    std::call_once(thh_init,[&] {
      thh_state = detail::getHIPHooks().initHIP();
    });
    return thh_state.get();
  }
  static const at::cuda::NVRTC& getNVRTC() {
    return detail::getCUDAHooks().nvrtc();
  }
  THCState* getTHCState() {
    // AT_ASSERT(thc_state);
    return thc_state.get();
  }
  THHState* getTHHState() {
    return thh_state.get();
  }

  static bool setFlushDenormal(bool on);

  // NB: This method is *purely* whether or not a user requested
  // that CuDNN was enabled, it doesn't actually say anything about
  // whether or not CuDNN is actually usable.  Use cudnn_is_acceptable
  // to test this instead
  bool userEnabledCuDNN() const;
  void setUserEnabledCuDNN(bool e);
  bool userEnabledMkldnn() const;
  void setUserEnabledMkldnn(bool e);
  bool benchmarkCuDNN() const;
  void setBenchmarkCuDNN(bool);
  bool deterministicCuDNN() const;
  void setDeterministicCuDNN(bool);

  // Note [Enabling Deterministic Operations]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Operations in PyTorch that normally act nondeterministically, but have an alternate
  // deterministic implementation, should satisfy the following requirements:
  //
  // * Include this comment: "See Note [Enabling Deterministic Operations]"
  //
  // * Check the value of `at::globalContext().deterministicAlgorithms()` to toggle
  //   between nondeterministic and deterministic implementations.
  //
  // * Have an entry in the list of PyTorch operations that toggle between nondeterministic
  //   and deterministic implementations, in the docstring of `use_deterministic_algorithms()`
  //   in torch/__init__.py
  //
  // `example_func()` below shows an example of toggling between nondeterministic and
  // deterministic implementations:
  //
  //    void example_func() {
  //      // See Note [Enabling Deterministic Operations]
  //      if (at::globalContext().deterministicAlgorithms()) {
  //        example_func_deterministic();
  //      } else {
  //        example_func_nondeterministic();
  //      }
  //    }

  bool deterministicAlgorithms() const;
  void setDeterministicAlgorithms(bool);

  // Note [Writing Nondeterministic Operations]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Operations in PyTorch that act nondeterministically and do not have an alternate
  // deterministic implementation should satisfy the following requirements:
  //
  // * Include this comment: "See Note [Writing Nondeterministic Operations]"
  //
  // * Include a comment explaining why the operation is nondeterministic.
  //
  // * Throw an error when `Context::deterministicAlgorithms()` is true. Most
  //   of the time, this should be accomplished by calling
  //   `at::globalContext().alertNotDeterminstic()`.  However, if the
  //   nondeterministic behavior is caused by the CuBLAS workspace
  //   configuration in CUDA >= 10.2,
  //   `at::globalContext().alertCuBLASConfigNotDeterministic()` should be
  //   called instead (in this case, a comment explaining why the operation is
  //   nondeterministic is not necessary). See below for details on these
  //   methods.
  //
  // * Have an entry in the list of nondeterministic PyTorch operations in the
  //   docstring of `use_deterministic_algorithms()` in torch/__init__.py
  //
  // * Have a test function in `test/test_torch.py` whose name begins with
  //   `test_nondeterministic_alert_`. Alternatively, if CuBLAS workspace
  //   configuration is the reason for nondeterminism, the operation should be
  //   included in the `test_cublas_config_nondeterministic_alert` test. Any new
  //   tests should ideally follow a pattern similar to the existing ones.
  //
  // `example_func()` below shows an example of the comments and error-throwing code
  // for a nondeterministic operation:
  //
  //    void example_func() {
  //      // See Note [Writing Nondeterministic Operations]
  //      // Nondeterministic because <reason>
  //      at::globalContext().alertNondeterministic("example_func");
  //      ...
  //    }

  // Throws an error if `Context::deterministicAlgorithms()` is true
  static void alertNotDeterministic(c10::string_view const& caller);

  // Throws an error if `Context::deterministicAlgorithms()` is true, CUDA >= 10.2, and
  // CUBLAS_WORKSPACE_CONFIG is not set to either ":16:8" or ":4096:8". For more details:
  // https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
  void alertCuBLASConfigNotDeterministic() const;

  bool allowTF32CuDNN() const;
  void setAllowTF32CuDNN(bool);
  bool allowTF32CuBLAS() const;
  void setAllowTF32CuBLAS(bool);
  at::QEngine qEngine() const;
  void setQEngine(at::QEngine e);
  static const std::vector<at::QEngine>& supportedQEngines() ;
  static bool isXNNPACKAvailable() ;
  // This method is used to release the original weight after pre-packing.
  // It should be called once before loading/running the model.
  // NB: By default it is set to true for mobile builds.
  void setReleaseWeightsWhenPrepacking(bool e);
  bool releaseWeightsWhenPrepacking() const;

  void setDisplayVmapFallbackWarnings(bool enabled);
  bool areVmapFallbackWarningsEnabled() const;

  void setDefaultMobileCPUAllocator();
  void unsetDefaultMobileCPUAllocator();

 private:
  void initCUDAIfNeeded(DeviceType p) {
    if (p == DeviceType::CUDA) {
      lazyInitCUDA();
    }
  }
  void initHIPIfNeeded(DeviceType p) {
    if (p == DeviceType::HIP) {
      lazyInitHIP();
    }
  }
  static bool checkCuBLASConfigDeterministic();
  std::once_flag thc_init;
  std::once_flag thh_init;
  bool enabled_cudnn = true;
  bool deterministic_cudnn = false;
  bool _deterministic_algorithms = false;
  bool benchmark_cudnn = false;
  bool allow_tf32_cudnn = true;
  bool allow_tf32_cublas = true;
  bool enabled_mkldnn = true;
  #ifdef C10_MOBILE
  bool release_original_weights = true;
  #else
  bool release_original_weights = false;
  #endif
  bool display_vmap_fallback_warnings_ = false;
  c10::optional<at::QEngine> quantized_engine = c10::nullopt;
  std::unique_ptr<THCState, void(*)(THCState*)> thc_state;
  std::unique_ptr<THHState, void(*)(THHState*)> thh_state;

  Allocator* prev_allocator_ptr_{nullptr};
};

TORCH_API Context& globalContext();

static inline void init() {
  globalContext();
}

TORCH_API Allocator* getCPUAllocator();

static inline DeprecatedTypeProperties& getDeprecatedTypeProperties(Backend p, ScalarType s) {
  return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
      p, s);
}

static inline DeprecatedTypeProperties& CPU(ScalarType s) {
  return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
      Backend::CPU, s);
}

static inline DeprecatedTypeProperties& CUDA(ScalarType s) {
  return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
      Backend::CUDA, s);
}

static inline DeprecatedTypeProperties& HIP(ScalarType s) {
  return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
      Backend::HIP, s);
}

static inline bool hasCUDA() {
  return globalContext().hasCUDA();
}

static inline bool hasHIP() {
  return globalContext().hasHIP();
}

static inline bool hasXLA() {
  return globalContext().hasXLA();
}

static inline bool hasMLC() {
  return globalContext().hasMLC();
}

// Despite its name, this function returns the number of *CUDA* GPUs.
static inline size_t getNumGPUs() {
  // WARNING: DO NOT ADD LOGIC TO HANDLE OTHER DEVICE TYPES TO THIS
  // FUNCTION.  If you are interested in interrogating the number of
  // devices for a specific device type, add that function to the
  // relevant library (e.g., similar to at::cuda::device_count())
  if (hasCUDA() && hasHIP()) {
    throw std::runtime_error(
        "Enabling both CUDA and HIP in ATen is not supported, as HIP masquerades "
        "to be CUDA (e.g., when you say CUDA, on a HIP build of ATen, this actually "
        "means HIP.  Rebuild PyTorch with one or the other disabled.");
  } else if (hasCUDA()) {
    return detail::getCUDAHooks().getNumGPUs();
  } else if (hasHIP()) {
    return detail::getHIPHooks().getNumGPUs();
  } else {
    return 0;
  }
}

static inline bool hasOpenMP() {
  return globalContext().hasOpenMP();
}

static inline bool hasMKL() {
  return globalContext().hasMKL();
}

static inline bool hasLAPACK() {
  return globalContext().hasLAPACK();
}

static inline bool hasMAGMA() {
  return globalContext().hasMAGMA();
}

static inline bool hasMKLDNN() {
  return globalContext().hasMKLDNN();
}

static inline void manual_seed(uint64_t seed) {
  auto gen = globalContext().defaultGenerator(DeviceType::CPU);
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen.mutex());
    gen.set_current_seed(seed);
  }
  // NB: Sometimes we build with CUDA, but we don't have any GPUs
  // available. In that case, we must not seed CUDA; it will fail!
  const auto num_gpus = detail::getCUDAHooks().getNumGPUs();
  if (hasCUDA() && num_gpus > 0) {
    for (int i = 0; i < num_gpus; i++) {
      auto cuda_gen = globalContext().defaultGenerator(
        Device(at::kCUDA, static_cast<c10::DeviceIndex>(i))
      );
      {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(cuda_gen.mutex());
        cuda_gen.set_current_seed(seed);
      }
    }
  }
}

// When the global flag `allow_tf32` is set to true, cuBLAS handles are
// automatically configured to use math mode CUBLAS_TF32_TENSOR_OP_MATH.
// For some operators, such as addmv, TF32 offers no performance improvement
// but causes precision loss. To help this case, this class implements
// a RAII guard that can be used to quickly disable TF32 within its scope.
//
// Usage:
//     NoTF32Guard disable_tf32;
struct TORCH_API NoTF32Guard {
  NoTF32Guard();
  ~NoTF32Guard();
  static bool should_disable_tf32();
private:
  bool changed = false;
};

} // namespace at
