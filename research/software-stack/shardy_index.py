"""Pinned Shardy integration and matmul propagation source anchors."""
LAYER = "Shardy round trip and partitioning"
SITES = [
    ('jax.sharding-representation','jax','jax/_src/interpreters/mlir.py','def _to_physical_op_sharding(',1202,'aval、sharding 与 axis context','SdyArray 或 OpSharding','配置决定前端 sharding 表示；extended dtype/manual axes 先处理。'),
    ('jax.compile-partitioner','jax','jax/_src/compiler.py','def get_compile_options(',180,'replicas/partitions、device assignment、配置','CompileOptions','use_spmd_partitioning=True；use_shardy_partitioner 来自 JAX 配置，不代表传播一定发生。'),
    ('xla.cpu-spmd-gate','xla','xla/service/cpu/cpu_compiler.cc','absl::Status CpuCompiler::RunHloPassesThroughLayoutAssn(',596,'HLO config 与 CPU options','分片传播/分区或单分区清理','num_partitions>1 时传播再 StatefulRngSpmdPartitioner；单分区构造 ShardyXLA(false)。'),
    ('xla.gspmd-detection','xla','xla/service/spmd/shardy/utils.cc','bool hasGspmdAttrsOrOps(mlir::ModuleOp module) {',360,'可能混有旧式 sharding 的 module','是否触发 GSPMD 兼容判断','非 main 参数、func results 和 Sharding custom calls 等有具体条件；不以任意 mhlo.sharding 存在就回退。'),
    ('xla.sdy-export-for-gspmd','xla','xla/pjrt/mlir_to_hlo.cc','absl::Status ExportShardyForGSPMD(mlir::ModuleOp module) {',231,'含 Shardy mesh 的 module','保留约束的 StableHLO','无 mesh 提前返回；用于 GSPMD 兼容，区别于保存 SDY 信息的 round-trip export。'),
    ('xla.sdy-roundtrip-export','xla','xla/service/spmd/shardy/sdy_round_trip/pipelines.cc','void addSdyRoundTripExportPipeline(mlir::OpPassManager& pm,',52,'SDY/StableHLO、mesh/V3 选项','可转 HLO 且保存 SDY 信息的 pipeline','lift/dedup、ops、shard_map、attrs 和 StableHLO shardings 各有步骤。'),
    ('xla.sdy-roundtrip-import','xla','xla/service/spmd/shardy/sdy_round_trip/pipelines.cc','void addSdyRoundTripImportPipeline(mlir::OpPassManager& pm,',72,'HLO 导回的 StableHLO 与隐藏属性','恢复 SDY 的 pipeline','常量转为 sdy.constant 防止后续折叠；恢复 mesh、attrs、custom calls、shard_map。'),
    ('xla.sdy-attrs-export','xla','xla/service/spmd/shardy/sdy_round_trip/export_shardy_attrs.cc','class SdyRoundTripExportShardyAttrsPass',150,'SDY sharding/rules/meshes','隐藏 frontend attrs 或 V3 表示','V3 开启时 frontend attrs 只存 rules；本轮 CPU 样本观察到非 V3 的 xla.sdy.meshes/sharding。'),
    ('xla.sdy-attrs-import','xla','xla/service/spmd/shardy/sdy_round_trip/import_shardy_attrs.cc','class SdyRoundTripImportShardyAttrsPass',575,'隐藏 attrs 与 V3 开关','恢复 mesh/tuple shardings 与 op attributes','恢复后的表示不能据函数名认为图或全部 HLO metadata 字节不变。'),
    ('xla.shardy-pass','xla','xla/service/spmd/shardy/shardy_xla_pass.cc','absl::StatusOr<bool> ShardyXLA::RunImpl(',470,'HloModule、propagation 选项','替换 computation 后的 HLO','可提前返回；执行往返时保存/恢复 entry layout、alias、donor config。非完整模块直接替换。'),
    ('xla.shardy-propagate','xla','xla/service/spmd/shardy/shardy_xla_pass.cc','absl::Status runShardingPropagation(HloModule* hloModule,',318,'从 HLO 导入的 MLIR 与 options','SDY import/propagation/export 结果','production 使用 SDY round-trip import；importMhloShardings 分支仅测试。V3 与旧 frontend payload 有独立 fallback。'),
    ('xla.shardy-replace-computations','xla','xla/service/spmd/shardy/shardy_xla_pass.cc','absl::Status createFromProtoAndReplaceComputations(',97,'转换后的 HloModuleProto 与原 HloModule','重建/替换 computations 并 DCE','名称/ID 可重新分配；不能用逐字文本相等作为往返语义保留的唯一判断。'),
    ('xla.shardy-dump-gate','xla','xla/service/spmd/shardy/shardy_xla_pass.cc','std::string getShardyDirIfShouldDump(const DebugOptions& debugOptions,',303,'dump 目录、pass regex、verbose','Shardy dump 根目录或空串','需 xla_dump_to 与匹配 pass；详细 Shardy 文件由内部保存点产生，不等于所有 pass 逐项 dump。'),
    ('xla.sdy-final-export','xla','xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_export.cc','void addStablehloExportPipeline(mlir::OpPassManager& pm,',31,'传播后 SDY module 和 export options','可供后续 HLO/SPMD 的 StableHLO','sdy.constant→stablehlo.constant、ops、shard_map、shardings、callbacks；不是最初保存 round-trip 信息的 export。'),
    ('shardy.import-pipeline','shardy','shardy/dialect/sdy/transforms/import/import_pipeline.cc','void addImportPipeline(OpPassManager& pm, int& dumpIndex,',30,'SDY module 与 propagation options','清理、dataflow edges 与 constraints','before_propagation dump 在部分清理之后，不等于最初输入 module。'),
    ('shardy.user-priority','shardy','shardy/dialect/sdy/transforms/propagation/user_priority_propagation.cc','LogicalResult UserPriorityPropagationPassImpl::propagate(',231,'sharding references 与用户优先级','按优先级运行传播','先 priority 0 再逐个用户优先级；本轮未添加不同用户优先级实验。'),
    ('shardy.op-priority','shardy','shardy/dialect/sdy/transforms/propagation/op_priority_propagation.cc','LogicalResult OpPriorityPropagationPassImpl::propagate(',178,'module 与方向规则','按 op schedule 的传播','配置可直接用 aggressive propagation；源码层次不等于本例执行次数采样。'),
    ('shardy.dot-general-rule','shardy','shardy/dialect/sdy/transforms/propagation/op_sharding_rule_registry.cc','      .Case([](stablehlo::DotGeneralOp dotGeneral) {',739,'batch/contracting/noncontracting 维度','matmul factor rule','batch 保留两边/输出，收缩维度标为 reduction；不能据允许分片推断最优设备划分。'),
    ('shardy.dot-rule','shardy','shardy/dialect/sdy/transforms/propagation/op_sharding_rule_registry.cc','      .Case([](stablehlo::DotOp dot) {',790,'向量或矩阵 dot 的输入 shape','factor rule','本例导回 StableHLO 为 dot；观察到 ([i,k],[k,j])->([i,j])，k 为 reduction。'),
    ('shardy.export-pipeline','shardy','shardy/dialect/sdy/transforms/export/export_pipeline.cc','void addExportPipeline(OpPassManager& pm, int& dumpIndex,',89,'传播后 SDY 与 export options','关闭 sharding、移除辅助信息、可选 reshards/collectives','after_propagation 保存点在清理之后；后续 minimal partitioner 仍使用 global shapes。'),
    ('shardy.minimal-partitioner','shardy','shardy/dialect/sdy/transforms/export/export_pipeline.cc','void runShardyPartitioner(OpPassManager& pm, int& dumpIndex,',39,'传播后的 constraints 与 export options','显式 reshard 或选定 collective/局部 partitioning','默认 enableInsertExplicitCollectives=false、enablePerInstructionPartitioning=false；本例真正 global→local 在后续 XLA SPMD。'),
]
SITES=[(i,c,LAYER,p,a,line,inp,out,limit) for i,c,p,a,line,inp,out,limit in SITES]
IDS={s[0] for s in SITES}|{'shardy.propagation','xla.mlir-to-hlo'}
EDGES=[
    ('xla.mlir-to-hlo','xla.gspmd-detection','        xla::sdy::hasGspmdAttrsOrOps(module)) {','direct','build options 开启 Shardy；混合旧式 IR 兼容检查。'),
    ('xla.mlir-to-hlo','xla.sdy-export-for-gspmd','      ABSL_RETURN_IF_ERROR(ExportShardyForGSPMD(module));','direct','检测到需 GSPMD 的 IR 后先关闭 Shardy/V3。'),
    ('xla.mlir-to-hlo','xla.sdy-roundtrip-export','    xla::sdy::addSdyRoundTripExportPipeline(pm, /*keepMeshesInlined=*/false,','direct','不按 use_shardy flag 跳过，纯 StableHLO 时可无 SDY 操作可处理。'),
    ('xla.cpu-spmd-gate','xla.shardy-pass','      spmd_pipeline.AddPass<sdy::ShardyXLA>();','pass registration','num_partitions>1 且 use_shardy=true。'),
    ('xla.cpu-spmd-gate','xla.shardy-pass','      sharding_removal_pipeline.AddPass<sdy::ShardyXLA>(','pass registration','num_partitions<=1，显式 runSdyShardingPropagation=false。'),
    ('xla.shardy-pass','xla.shardy-propagate','    ABSL_RETURN_IF_ERROR(runShardingPropagation(','direct','runSdyShardingPropagation=true。'),
    ('xla.shardy-pass','xla.shardy-replace-computations','      createFromProtoAndReplaceComputations(hloModule, hloProto.hlo_module()));','direct','完成 StableHLO→HLO 转换后。'),
    ('xla.shardy-propagate','xla.shardy-dump-gate','      getShardyDirIfShouldDump(debugOptions, passName, isShardyVerbose);','direct',''),
    ('xla.shardy-propagate','xla.sdy-roundtrip-import','    addSdyRoundTripImportPipeline(pm, /*enableConstantImport=*/true,','direct','production 分支；测试 importMhloShardings=false。'),
    ('xla.shardy-propagate','shardy.propagation','  mlir::sdy::addPropagationPipeline(pm, dumpIndex, options);','direct','完成 import 后。'),
    ('xla.shardy-propagate','xla.sdy-final-export','  addStablehloExportPipeline(pm, stablehloExportPipelineOptions);','direct','SDY propagation/export 后。'),
    ('shardy.propagation','shardy.import-pipeline','  addImportPipeline(pm, dumpIndex, options);','direct',''),
    ('shardy.propagation','shardy.export-pipeline','  addExportPipeline(pm, dumpIndex, exportOptions);','direct','传播和函数 call 处理后。'),
    ('shardy.user-priority','shardy.op-priority','  if (failed(OpPriorityPropagationPassImpl::propagate(','direct','首先 priority 0。'),
    ('shardy.export-pipeline','shardy.minimal-partitioner','    runShardyPartitioner(pm, dumpIndex, options);','direct','avoidExportForPartitioning=false。'),
]
