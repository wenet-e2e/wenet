import torch
import poptorch
import os


class IpuOptionBuilder:
    def __init__(self, config):
        self.config = config
        torch.manual_seed(config['seed'])

    def build_train_ipu_options(self):
        options = self._build_core_ipu_options()
        options.Training.gradientAccumulation(
            self.config["gradient_accumulation"])
        options.deviceIterations(self.config['device_iterations'])
        options.outputMode(poptorch.OutputMode.Final)
        if self.config['enable_profile']:
            if not os.path.exists(self.config['profile_path']):
                os.mkdir(self.config['profile_path'])
            options._Popart.set("engineOptions", self.engine_options)
        return options

    def build_validate_ipu_options(self):
        options = self._build_core_ipu_options()
        options.Training.gradientAccumulation(1)
        options.deviceIterations(self.config['validate_device_iterations'])
        options.outputMode(poptorch.OutputMode.All)
        return options

    def _build_core_ipu_options(self):
        options = poptorch.Options()
        options.randomSeed(self.config['seed'])
        options.autoRoundNumIPUs(True)
        options.replicationFactor(self.config["num_replicas"])
        options.enableExecutableCaching(self.config["executable_cache"])
        options.Training.accumulationAndReplicationReductionType(
            poptorch.ReductionType.Mean)
        options.setExecutionStrategy(
            poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))
        options.TensorLocations.setOptimizerLocation(
            self.tensor_location_settings)
        if self.config["enable_half_partials"]:
            options.Precision.setPartialsType(torch.half)
        amp = self.config["available_memory_propotion"]
        options.setAvailableMemoryProportion(
            {f'IPU{i}': amp_ for i, amp_ in enumerate(amp)})
        options.Precision.enableStochasticRounding(
            self.config["enable_stochastic_rounding"])
        # Popart settings
        # enable recomputation in pipeline mode
        options._Popart.set('autoRecomputation', 3)
        options._Popart.set('disableGradAccumulationTensorStreams', True)
        options._Popart.set('outlineThreshold', 10.0)
        options._Popart.set(
            'accumulateOuterFragmentSettings.excludedVirtualGraphs', ['0'])
        options._Popart.set(
            'scheduleNonWeightUpdateGradientConsumersEarly', True)
        options._Popart.setPatterns({
            'TiedGather': True,
            'TiedGatherAccumulate': True,
            'UpdateInplacePrioritiesForIpu': True
        })
        return options

    @property
    def tensor_location_settings(self):
        tensor_location_settings = poptorch.TensorLocationSettings()
        tensor_location_settings.useOnChipStorage(
            not self.config["optimizer_state_offchip"])
        tensor_location_settings.useReplicatedTensorSharding(
            self.config["num_replicas"] > 1)
        return tensor_location_settings

    @property
    def engine_options(self):
        engine_options = {
            "opt.useAutoloader": "true",
            "opt.maxCompilationThreads": 40,
            "autoReport.directory": self.profile_folder_name,
            "target.syncReplicasIndependently": "true",
            "debug.allowOutOfMemory": "true",
            "profiler.format": "v3",
            "profiler.includeFlopEstimates": "true",
            "profiler.includeCycleEstimates": "true",
            "autoReport.all": "true",
            "autoReport.executionProfileProgramRunCount": "2",
        }
        return engine_options

    @property
    def profile_folder_name(self):
        prefix = self.config['profile_prefix']
        lbs = self.config['local_batch_size']
        ga = self.config['gradient_accumulation']
        replica = self.config['num_replicas']
        sdk_version = os.environ.get(
            "POPLAR_SDK_ENABLED").split("/")[-2].split("-")[-3:]
        sdk_version = "-".join(sdk_version)
        pipeline = ','.join(i[0] for i in self.config['pipeline'])
        amp_list = self.config['available_memory_propotion']
        amp = ','.join([str(i) for i in amp_list])
        folder_name = f'{prefix}_lbs{lbs}_ga{ga}_rep{replica}_pipe{pipeline}_amp{amp}_sdk{sdk_version}'
        folder_name = os.path.join(self.config['profile_path'], folder_name)
        return folder_name
