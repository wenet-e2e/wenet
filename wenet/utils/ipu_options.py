import torch
import poptorch
import popart
import os



def _build_ipu_option(config):
    opts = poptorch.Options()
    torch.manual_seed(config["seed"])
    opts.randomSeed(config["seed"])
    replicas = config["num_replicas"]
    amp = config["available_memory_propotion"]
    if isinstance(amp, float):
        amp = [amp for _ in range(len(config["pipeline"]) + 1)]
    opts.setAvailableMemoryProportion({f'IPU{i}': amp_ for i,amp_ in enumerate(amp)})
    opts.replicationFactor(replicas)
    opts.autoRoundNumIPUs(True)
    
    if config["enable_half_partials"]:
        opts.Precision.setPartialsType(torch.half)
    opts.Precision.enableStochasticRounding(config["enable_stochastic_rounding"])
    opts.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Mean)

    opts.setExecutionStrategy(poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))
    opts.TensorLocations.setOptimizerLocation(
                poptorch.TensorLocationSettings()
                .useOnChipStorage(not config["optimizer_state_offchip"])
                .useReplicatedTensorSharding(replicas>1)
            )
    opts.enableExecutableCaching(config["executable_cache"])
    # Popart settings
    opts._Popart.set('autoRecomputation', 3)
    opts._Popart.set('disableGradAccumulationTensorStreams', True)
    opts._Popart.set('outlineThreshold', 10.0)
    opts._Popart.set('accumulateOuterFragmentSettings.excludedVirtualGraphs', ['0'])
    opts._Popart.set('subgraphCopyingStrategy', int(popart.SubgraphCopyingStrategy.JustInTime))
    opts._Popart.set('scheduleNonWeightUpdateGradientConsumersEarly', True)
    opts._Popart.setPatterns({
        'TiedGather': True,
        'TiedGatherAccumulate': True,
        'UpdateInplacePrioritiesForIpu': True
    })
    return opts


def build_ipu_option(config):
    ga = config["gradient_accumulation"]
    opts = _build_ipu_option(config)
    opts.Training.gradientAccumulation(ga)
    opts.deviceIterations(config['device_iterations'])
    opts.outputMode(poptorch.OutputMode.Final)
    lbs = config["local_batch_size"]
    ga = config["gradient_accumulation"]
    replicas = config["num_replicas"]
    amp = config["available_memory_propotion"]
    if isinstance(amp, float):
        amp = [amp for _ in range(len(config["pipeline"]) + 1)]
    if config["enable_profile"]:
        ampstr = ','.join([str(amp_) for amp_ in amp])
        sdk_version = "-".join(os.environ.get("POPLAR_SDK_ENABLED").split("/")[-2].split("-")[-3:])
        profile_path = os.path.join(
            os.path.abspath(config["profile_path"]),
            f"bs{lbs}-ga{ga}-amp{ampstr}-rep{replicas}-{sdk_version}")

        if not os.path.exists(profile_path):
            os.makedirs(profile_path)
        engine_options = {
            "opt.useAutoloader": "true",
            "opt.maxCompilationThreads": 40,
            "autoReport.directory": profile_path,
            "target.syncReplicasIndependently": "true",
            # "opt.internalExchangeOptimisationTarget": "memory",
            "debug.allowOutOfMemory": "true",
            "profiler.format": "v3",
            "profiler.includeFlopEstimates": "true",
            "profiler.includeCycleEstimates": "true",
            "autoReport.all": "true",
            "autoReport.executionProfileProgramRunCount": "2",
        }
        opts._Popart.set("engineOptions", engine_options)
    return opts


def build_ipu_option_validate(config):
    opts = _build_ipu_option(config)
    opts.Training.gradientAccumulation(1)
    opts.deviceIterations(16)
    opts.outputMode(poptorch.OutputMode.All)
    return opts
