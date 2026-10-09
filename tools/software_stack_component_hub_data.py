"""Pinned-source content for the software-stack component hub diagrams."""

from render_overview_software_stack_components import SOURCE_METADATA

PINS = dict(SOURCE_METADATA['source_pins'], **{
    'llvm-project': '75a45c373407c13a44c7abb28a78d891a97fe665',
    'stablehlo': '7b1b15781ccbd770f50c7eef4b0c3e03834649fd',
})
REPOSITORIES = {'jax': '0xaskr/jax', 'xla': 'openxla/xla',
                'llvm-project': 'llvm/llvm-project', 'stablehlo': 'openxla/stablehlo'}
SOURCES = {name: (item['repo'], item['path'], item['needle'], item['line'])
           for name, item in SOURCE_METADATA['source_anchors'].items()}


def anchor(name, repo, path, needle, line=None):
    SOURCES[name] = (repo, path, needle, line)
    return name


JMLIR = 'jax/_src/interpreters/mlir.py'
JLIB = 'jaxlib/mlir.cc'
IFRT = 'xla/python/ifrt/'
PIFRT = 'xla/python/pjrt_ifrt/'
PJRT = 'xla/pjrt/pjrt_client.h'
CPU = 'xla/service/cpu/cpu_compiler.cc'
GPU = 'xla/service/gpu/gpu_compiler.cc'
IRC = 'xla/backends/cpu/codegen/ir_compiler.cc'
TRITON = 'xla/backends/gpu/codegen/triton/'
SE = 'xla/stream_executor/'
CAPI = 'xla/pjrt/c/pjrt_c_api.h'

for name, repo, path, needle in [
    ('ModuleOp', 'llvm-project', 'mlir/include/mlir/IR/BuiltinOps.td', 'def ModuleOp :'),
    ('StableHLO ops', 'stablehlo', 'stablehlo/dialect/StablehloOps.td', 'def StableHLO_AddOp :'),
    ('ModuleContext', 'jax', JMLIR, 'class ModuleContext:'),
    ('MLIR verify', 'jax', JMLIR, 'if not ctx.module.operation.verify():'),
    ('Shardy module pass', 'jax', JMLIR, "'builtin.module(sdy-lift-inlined-meshes)'"),
    ('module_to_string', 'jax', JMLIR, 'def module_to_string('),
    ('module_to_bytecode', 'jax', JMLIR, 'def module_to_bytecode('),
    ('HLO to Module', 'jax', JLIB, 'absl::StatusOr<std::string> PyXlaComputationToMlirModule('),
    ('MHLO to StableHLO', 'jax', JLIB, 'absl::StatusOr<nb::bytes> PyMhloToStablehlo('),
    ('deserialize artifact', 'jax', JLIB, 'absl::StatusOr<nb::object> PyDeserializePortableArtifact('),
    ('ArraySpec', 'xla', IFRT+'array_spec.h', 'struct ArraySpec {'),
    ('Sharding', 'xla', IFRT+'sharding.h', 'class Sharding :'),
    ('Value', 'xla', IFRT+'value.h', 'class Value :'),
    ('PjRtArray', 'xla', PIFRT+'pjrt_array.h', 'class PjRtArray final :'),
    ('Assemble arrays', 'xla', PIFRT+'pjrt_client.cc', 'PjRtClient::AssembleArrayFromSingleDeviceArrays('),
    ('Bitcast arrays', 'xla', PIFRT+'pjrt_client.cc', 'PjRtClient::BitcastArrays('),
    ('Disassemble arrays', 'xla', PIFRT+'pjrt_array.cc', 'PjRtArray::DisassembleIntoSingleDeviceArrays('),
    ('Array to host', 'xla', PIFRT+'pjrt_array.cc', 'PjRtArray::CopyToHostBuffer('),
    ('Array readiness', 'xla', PIFRT+'pjrt_array.cc', 'PjRtArray::GetReadyFuture() const'),
    ('Array delete', 'xla', PIFRT+'pjrt_array.cc', 'PjRtArray::Delete() {'),
    ('Buffer definition', 'xla', PJRT, 'class PjRtBuffer {'),
    ('Buffer bitcast', 'xla', PJRT, 'virtual absl::StatusOr<std::unique_ptr<PjRtBuffer>> Bitcast('),
    ('Buffer dependency', 'xla', PJRT, 'DonateWithControlDependency(Future<> dependency)'),
    ('Buffer release', 'xla', PJRT, 'ReleaseDeviceMemoryOwnership(bool wait_for_operations_to_complete) = 0;'),
    ('Buffer delete', 'xla', PJRT, 'virtual void Delete() = 0;'),
    ('Buffer to literal', 'xla', PJRT, 'virtual Future<> ToLiteral(MutableLiteralBase* literal) = 0;'),
    ('Buffer external ref', 'xla', PJRT, 'AcquireExternalReference() = 0;'),
    ('PJRT Execute', 'xla', PJRT, 'Execute(absl::Span<const std::vector<PjRtBuffer*>> argument_handles,'),
    ('HloComputation', 'xla', 'xla/hlo/ir/hlo_computation.h', 'class HloComputation {'),
    ('HloInstruction', 'xla', 'xla/hlo/ir/hlo_instruction.h', 'class HloInstruction {'),
    ('HloVerifier', 'xla', 'xla/service/hlo_verifier.h', 'class HloVerifier :'),
    ('AlgebraicSimplifier', 'xla', 'xla/hlo/transforms/simplifiers/algebraic_simplifier.h', 'class AlgebraicSimplifier :'),
    ('HloDCE', 'xla', 'xla/hlo/transforms/simplifiers/hlo_dce.h', 'class HloDCE :'),
    ('LLVM Module', 'llvm-project', 'llvm/include/llvm/IR/Module.h', 'class LLVM_ABI Module {'),
    ('LLVM Function', 'llvm-project', 'llvm/include/llvm/IR/Function.h', 'class LLVM_ABI Function :'),
    ('LLVM BasicBlock', 'llvm-project', 'llvm/include/llvm/IR/BasicBlock.h', 'class BasicBlock final :'),
    ('LLVM Instruction', 'llvm-project', 'llvm/include/llvm/IR/Instruction.h', 'class Instruction :'),
    ('CPU IrEmitter2', 'xla', CPU, 'IrEmitter2 ir_emitter2('),
    ('CPU ThunkEmitter', 'xla', CPU, 'thunk_emitter.EmitEntryComputation(*module)'),
    ('CPU IR verification', 'xla', CPU, 'absl::Status VerifyLlvmModule('),
    ('CPU unused symbols', 'xla', CPU, 'static void RemoveUnusedSymbols('),
    ('CPU extra kernels', 'xla', CPU, 'static absl::StatusOr<std::unique_ptr<llvm::Module>> ExtractKernelsFromModule('),
    ('CPU split module', 'xla', CPU, 'llvm::SplitModule(*llvm_module,'),
    ('CPU IR passes', 'xla', IRC, 'llvm::Error IrCompiler::RunIrPasses('),
    ('CPU machine code', 'xla', IRC, 'std::unique_ptr<llvm::MemoryBuffer> IrCompiler::EmitMachineCode('),
    ('CPU IR compiler', 'xla', IRC, 'IrCompiler::operator()('),
    ('GPU module emission', 'xla', 'xla/service/gpu/compile_module_to_llvm_ir.cc', 'absl::StatusOr<CompileModuleResults> CompileModuleToLlvmIr('),
    ('Triton fusion', 'xla', TRITON+'fusion.cc', 'TritonFusion::GenerateTritonKernelAndWrapper('),
    ('Triton XLA pipeline', 'xla', TRITON+'compilation_pipeline.cc', 'void CreateTritonXlaPipeline('),
    ('Triton target pipeline', 'xla', TRITON+'compilation_pipeline.cc', 'void CreateTritonPipeline('),
    ('Triton to LLVM', 'xla', 'xla/backends/gpu/codegen/cubin_custom_kernel_compiler.cc', 'CubinCustomKernelCompiler::CompileTritonToLlvm('),
    ('GPU link and optimize', 'xla', 'xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.cc', 'absl::Status LinkAndOptimizeModule('),
    ('NVPTX target binary', 'xla', 'xla/service/gpu/nvptx_compiler.cc', 'NVPTXCompiler::CompileTargetBinary('),
    ('TPU plugin', 'jax', 'jax/_src/xla_bridge.py', 'def make_tpu_client('),
    ('PJRT Program', 'xla', CAPI, 'struct PJRT_Program {'),
    ('PJRT compile ABI', 'xla', CAPI, 'struct PJRT_Client_Compile_Args {'),
    ('PJRT execute ABI', 'xla', CAPI, 'struct PJRT_LoadedExecutable_Execute_Args {'),
    ('PJRT serialize ABI', 'xla', CAPI, 'struct PJRT_Executable_Serialize_Args {'),
    ('TPU opaque program', 'xla', 'xla/tpu/tpu_ops_c_api.h', 'typedef struct XLA_TpuProgram XLA_TpuProgram;'),
    ('C API program serialization', 'xla', 'xla/pjrt/c_api_client/pjrt_c_api_client.cc', 'absl::StatusOr<std::pair<std::string, std::string>> SerializeProgram('),
    ('C API compile call', 'xla', 'xla/pjrt/c_api_client/pjrt_c_api_client.cc', 'InitializeArgsAndCompile(PjRtCApiClient* api_client, const PJRT_Api* c_api,'),
    ('C API execute call', 'xla', 'xla/pjrt/c_api_client/pjrt_c_api_client.cc', 'PjRtCApiLoadedExecutable::Execute('),
    ('Mosaic to LLO design', 'jax', 'docs/pallas/design/design.md', 'Mosaic consumes (mostly) standard dialect MLIR and emits LLO to be'),
    ('TPU core classification', 'jax', 'jax/_src/tpu_custom_call.py', 'def _get_device_type('),
    ('TPU execution units', 'jax', 'docs/pallas/tpu/details.rst', 'In a nutshell, the main difference between TPUs and GPUs is that TPUs are'),
    ('LLO profiling metadata', 'xla', 'xla/tsl/profiler/utils/xplane_schema.cc', '{"llo_proto", kLloProto}'),
    ('HloSchedule', 'xla', 'xla/hlo/ir/hlo_schedule.h', 'class HloSchedule {'),
    ('LHS scheduling', 'xla', 'xla/service/latency_hiding_scheduler.cc', 'absl::StatusOr<bool> LatencyHidingScheduler::RunImpl('),
    ('Stream', 'xla', SE+'stream.h', 'class Stream {'),
    ('Stream launch', 'xla', SE+'stream.h', 'virtual absl::Status LaunchKernel('),
    ('Stream wait', 'xla', SE+'stream.h', 'virtual absl::Status WaitFor(Event* event) = 0;'),
    ('Stream record', 'xla', SE+'stream.h', 'virtual absl::Status RecordEvent(Event* event) = 0;'),
    ('Stream synchronize', 'xla', SE+'stream.h', 'virtual absl::Status BlockHostUntilDone() = 0;'),
    ('CUDA load kernel', 'xla', SE+'cuda/cuda_executor.cc', 'CudaExecutor::LoadKernel('),
    ('CUDA launch', 'xla', SE+'cuda/cuda_stream.cc', 'absl::Status CudaStream::LaunchKernel('),
    ('CUDA driver launch', 'xla', SE+'cuda/cuda_stream.cc', 'absl::Status LaunchCudaKernel('),
    ('CUDA graph capture', 'xla', SE+'cuda/cuda_stream.cc', 'CudaStream::CaptureHandle::BeginCapture('),
    ('CUDA synchronize', 'xla', SE+'cuda/cuda_stream.cc', 'absl::Status CudaStream::BlockHostUntilDone() {'),
    ('Kernel arguments', 'xla', 'xla/backends/gpu/runtime/kernel_thunk.cc', 'KernelThunk::GetKernelAndArgs('),
    ('Kernel submission', 'xla', 'xla/backends/gpu/runtime/kernel_thunk.cc', 'absl::Status KernelThunk::ExecuteOnStream('),
    ('Command buffer submission', 'xla', 'xla/backends/gpu/runtime/command_buffer_thunk.cc', 'absl::Status CommandBufferThunk::ExecuteOnStream('),
]:
    anchor(name, repo, path, needle)
anchor('serialize artifact', 'jax', JLIB, 'absl::StatusOr<nb::bytes> PySerializePortableArtifact(', 176)
anchor('PjRtArray Create', 'xla', PIFRT+'pjrt_array.cc', 'PjRtArray::Create(', 158)
anchor('Buffer delete', 'xla', PJRT, 'virtual void Delete() = 0;', 1273)
anchor('CUDA graph capture', 'xla', SE+'cuda/cuda_stream.cc', 'CudaStream::CaptureHandle::BeginCapture(', 200)
anchor('CUDA memcpy', 'xla', SE+'cuda/cuda_stream.cc', 'absl::Status AsynchronousMemcpyH2D(')
anchor('GPU single module', 'xla', GPU, 'GpuCompiler::CompileSingleModule(')
anchor('HLO from proto', 'xla', 'xla/hlo/ir/hlo_module.h', 'static absl::StatusOr<std::unique_ptr<HloModule>> CreateFromProto(', 570)
anchor('HLO entry computation', 'xla', 'xla/hlo/ir/hlo_module.h', 'HloComputation* AddEntryComputation(')
anchor('HLO clone', 'xla', 'xla/hlo/ir/hlo_module.h', 'std::unique_ptr<HloModule> Clone(')
anchor('HLO replace computations', 'xla', 'xla/hlo/ir/hlo_module.h', 'void ReplaceComputations(')
anchor('HLO text', 'xla', 'xla/hlo/ir/hlo_module.h', 'std::string ToString() const;', 505)
anchor('HLO proto', 'xla', 'xla/hlo/ir/hlo_module.h', 'void ToProto(HloModuleProto* proto,')
anchor('CPU AddModule', 'xla', CPU, 'llvm_module_compiler->AddModule(std::move(tsm), dylib_index++));')


def node(title, rows, *refs, tag=None, uncertain=False):
    return dict(title=title, rows=rows, refs=refs, tag=tag, uncertain=uncertain)


def consume(title, rows, result, result_rows, *refs, uncertain=False):
    return dict(node(title, rows, *refs, uncertain=uncertain), result=result,
                result_rows=result_rows)


DIAGRAMS = [
    dict(key='jaxlib', path='jaxlib/mlir-module-centered-hub.svg', title='jaxlib · 以 MLIR Module 为中心',
         core='MLIR Module（StableHLO）', core_ref='ModuleOp',
         summary='绑定层提供对象与操作接口；Module 是容器，StableHLO 是其中一种方言。',
         center=['Python ir.Module ↔ 原生 mlir::ModuleOp。', '主体：Region / Block / Operation；SSA Value 连接操作。',
                 '语义：操作名、类型、属性、符号及所用方言。', 'StableHLO、func、sdy 等可以共存于一个 Module。',
                 'jaxlib 提供绑定，JAX lowering 负责组织与构造。'],
         definitions=[
             node('ModuleOp / Region / Block', ['顶层容器具有单个区域和块，内部可放置操作。', 'SymbolTable 管理符号；IsolatedFromAbove 限制隐式捕获。', 'Module 的结构由 MLIR 定义。'], 'ModuleOp'),
             node('Operation / SSA / Type', ['操作的输入、结果、属性和嵌套区域描述计算。', 'StableHLO 的约束由方言操作定义及 verifier 承担。', 'Module 本身不等于 StableHLO 方言。'], 'StableHLO ops', 'ModuleOp'),
             node('Context / ModuleContext', ['Context 管理方言、类型和属性；Module 持有程序。', 'JAX ModuleContext 附加 lowering 规则与符号上下文。', 'Python 包装与原生对象的生命周期必须协调。'], 'ModuleContext'),
             node('验证与编译边界', ['operation.verify 检查 IR 的结构与语义约束。', '合法 Module 不等于已编译程序或可执行设备代码。', 'PyArray / IFRT Array 属于另一条数据通路。'], 'MLIR verify', 'PyClient CompileAndLoad'),
         ],
         producers=[
             node('JAX lowering 构造', ['输入：Jaxpr + avals / 平台 / 分片 / 效果信息。', 'ModuleContext 创建 ir.Module；lowering 规则填充操作。', '输出：LoweringResult.module；上下文另保留回调等。'], 'lower_jaxpr_to_module', 'ModuleContext'),
             node('反序列化 portable artifact', ['输入：版本化 StableHLO 字节 + MLIR Context。', 'PyDeserializePortableArtifact 调用方言反序列化。', '输出：包装后的 Module；无效字节返回错误。'], 'deserialize artifact'),
             node('HLO 兼容转换入口', ['输入：XlaComputation 的 HloModuleProto。', 'PyXlaComputationToMlirModule 调用 ConvertHloToStablehlo。', '内部生成 Module，外层接口输出模块文本。'], 'HLO to Module'),
         ],
         transforms=[
             node('clone / 编译隔离', ['输入：已有 Module。', 'PyClient::CompileAndLoad 先 clone，再允许后端原地修改。', '输出：新的 Module 对象；原 Python 输入得以保留。'], 'PyClient CompileAndLoad'),
             node('Shardy 模块内 pass', ['输入：带分片信息的 Module。', 'sdy-lift-inlined-meshes 在启用 Shardy 时运行。', '输出：修改后的 Module；此例不是所有 pass 的固定顺序。'], 'Shardy module pass'),
             node('MHLO → StableHLO', ['输入：含 MHLO 的模块文本 / 字节。', 'PyMhloToStablehlo 解析模块并运行合法化 pass。', '内部更新 Module，接口返回模块 bytecode。'], 'MHLO to StableHLO'),
             node('PrepareForExport', ['输入：交给 XLA 的 MLIR Module。', '合法化和 prepare-for-HLO-export 整理模块；动态形状按条件处理。', '输出：适合导出 HLO 的 Module；实现归 XLA。'], 'PrepareForExport'),
         ],
         after='变换后的 MLIR Module', after_rows=['仍使用 MLIR 的操作 / 区域 / 类型体系。', '可原地修改或 clone；不同方言和属性可被重写。'],
         consumers=[
             consume('PyClient::CompileAndLoad', ['读取 Module + CompileOptions。', '用 HloProgram 包装模块，提交 IFRT 编译。'], 'PyLoadedExecutable', ['编译结果包装回 Python；编译接口不接收执行数组。', 'HloProgram 名称不表示已经是 HloModule。'], 'PyClient CompileAndLoad'),
             consume('ConvertMlirHloToHloModule', ['读取 Module，经导出及配置转换。', '这是进入 XLA HLO 表示的接口。'], 'HloModule', ['产生 HLO 计算和指令结构。', 'Array / Buffer 不经此接口 lowering。'], 'ConvertMlirHloToHloModule'),
             consume('serialize_portable_artifact', ['读取 Module + 目标版本。', '版本化序列化走原生绑定。'], 'StableHLO 字节', ['用于传输、保存与后续反序列化。', '字节内容不是已加载设备可执行对象。'], 'serialize artifact'),
             consume('verify / 文本与 bytecode', ['读取 module.operation 或模块内容。', '校验、打印、普通 bytecode 是不同动作。'], '判定 / 文本 / 字节', ['校验失败产生诊断；打印和存储不执行计算。', '普通 MLIR bytecode 与版本化 artifact 区分。'], 'MLIR verify', 'module_to_bytecode'),
         ], foot='组件边界：jaxlib 是绑定与实现组件；MLIR / StableHLO 的语义定义来自对应上游，编译提交与数组执行分开。'),

    dict(key='ifrt', path='ifrt/array-centered-hub.svg', title='IFRT · 以逻辑 Array 为中心', core='IFRT Array', core_ref='ifrt Array',
         summary='一个逻辑数组可以分布在多个设备；以下具体实现采用 PJRT-backed PjRtArray。',
         center=['ArrayRef = RCReference<Array>；Array 继承 Value。', 'ArraySpec：dtype、shape、sharding、layout。',
                 'Sharding 关联设备列表、内存种类与分片规则。', 'PjRtArray 以一组 PjRtBuffer 支撑可寻址分片。',
                 '数组可在异步完成前返回；就绪状态由 Future 表达。'],
         definitions=[
             node('Array / ArraySpec', ['逻辑 shape 和 dtype 表示用户看到的一个数组。', 'ArraySpec 汇总静态属性，供输入 / 输出约束使用。', 'Array 本身不是程序 IR。'], 'ifrt Array', 'ArraySpec'),
             node('Sharding / DeviceList', ['devices + memory_kind 指定分片放置。', 'GetShardShape / IndexDomains 描述局部片段。', '全复制时每个分片可覆盖整个逻辑 shape。'], 'Sharding'),
             node('PjRtArray / PjRtBuffer', ['一个 Array 管理多个可寻址分片 Buffer。', '逻辑数组与物理存储是实现映射关系。', '此图的 PJRT-backed 细节不是所有 IFRT 实现的要求。'], 'PjRtArray'),
             node('复制、捐赠与就绪', ['ArrayCopySemantics：AlwaysCopy / ReuseInput / DonateInput。', '复用和捐赠决定底层 Buffer 的所有权行为。', 'Value 提供 GetReadyFuture / Delete 等生命周期接口。'], 'ifrt Array', 'Value'),
         ],
         producers=[
             node('MakeArrayFromHostBuffer', ['输入：主机地址 + dtype / shape / strides / sharding。', 'PJRT 实现支持单设备或全复制；按设备创建 Buffer。', '输出：PjRtArray；主机存储有效期遵守 HostBufferSemantics。'], 'MakeArrayFromHostBuffer'),
             node('AssembleArrayFromSingleDeviceArrays', ['输入：多个单设备 Array + 目标逻辑 shape / sharding。', '检查属性并按复制语义汇集分片 Buffer。', '输出：组合后的逻辑 Array；不是拼出 HLO。'], 'Assemble arrays'),
             node('LoadedExecutable 的输出包装', ['输入：PJRT Execute 返回的各设备结果 Buffer。', 'IFRT 按输出 dtype / shape / sharding 组织 PjRtArray。', '输出：ExecuteResult.outputs 中的 ArrayRef。'], 'IFRT PjRtLoadedExecutable Execute', 'PjRtArray Create'),
         ],
         transforms=[
             node('CopyArrays', ['输入：Array 列表 + 目标设备 / memory_kind + 复制语义。', '按实现能力选择本地或跨主机路径；设备数量受约束。', '输出：新的 Array 列表；可能复用、捐赠或复制 Buffer。'], 'CopyArrays'),
             node('RemapArrays', ['输入：Array 列表 + RemapPlan + 复制语义。', '按 mapping 将输入分片放到各输出的分片槽位。', '输出：重组后的 Array；不等同于任意数值重分片。'], 'RemapArrays'),
             node('DisassembleIntoSingleDeviceArrays', ['输入：一个逻辑 Array + 分片 / 复制语义。', '为可访问分片建立单设备数组表示。', '输出：多个 ArrayRef；逆向操作为 Assemble。'], 'Disassemble arrays'),
             node('BitcastArrays', ['输入：Array 列表 + 新 ArraySpec。', '当前 PJRT 实现要求 DonateInput，逐 Buffer 做 bitcast。', '输出：重解释后的 Array；原数组被删除，不是数值转换。'], 'Bitcast arrays'),
         ],
         after='新 Array / 多个 Array', after_rows=['核心抽象仍是逻辑数组；shape、放置或所有权按操作改变。', '这些是可选数据操作，不是每次执行都必经的阶段。'],
         consumers=[
             consume('LoadedExecutable::Execute', ['输入：ArrayRef 列表 + 执行选项。', 'PJRT 实现提取每个设备的 Buffer 参数。'], '输出 Array + 状态', ['执行输出包装为新的逻辑数组。', 'fill_status 启用时填充执行状态 Future。'], 'IFRT PjRtLoadedExecutable Execute'),
             consume('CopyToHostBuffer', ['读取 Array，并写入调用方提供的主机地址。', '受分片可寻址性、布局和实现支持限制。'], '主机数据 + Future', ['Future 成功后数据有效。', '在完成前必须保持主机目标地址有效。'], 'Array to host', 'ifrt Array'),
             consume('GetReadyFuture', ['读取 Array 对应 Buffer 的就绪状态。', '多 Buffer 时合并各自的 Future。'], '就绪 / 错误', ['就绪是数据依赖的完成状态。', '它不构造编译 IR，也不自动回传主机。'], 'Array readiness'),
             consume('Delete', ['释放 Array 对各底层 Buffer 的引用 / 使用资格。', '当前实现逐个调用 Buffer::Delete。'], '失效状态 / 生命周期结果', ['物理内存回收受未完成操作和共享引用约束。', '此版本 Array::Delete 的 Future 实现仍带 TODO。'], 'Array delete'),
         ], foot='Array → Buffer 是实现与存储映射；CopyArrays / RemapArrays / BitcastArrays 为按需路径，不能画成 Array lowering 为 HLO。'),

    dict(key='pjrt', path='pjrt/buffer-centered-hub.svg', title='PJRT · 以 Buffer 为中心', core='PjRtBuffer', core_ref='PjRtBuffer',
         summary='统一设备存储、就绪与所有权契约；实现由 CPU、GPU 或设备插件提供。',
         center=['PjRtBuffer 是设备数据句柄，接口由具体 provider 实现。', 'on_device_shape / layout / element_type / dimensions 描述数据。',
                 'client / device / memory_space 描述归属和放置。', 'GetReadyFuture 表示计算或传输何时完成。',
                 'PjRtLoadedExecutable 在执行时读取 Buffer 参数；编译接收程序。'],
         definitions=[
             node('Shape / Layout / 动态维度', ['Buffer 可以表示数组或元组。', '逻辑动态 shape 的查询可能涉及同步与设备通信。', 'on_device_shape 与逻辑有效形状需要区分。'], 'Buffer definition'),
             node('Client / Device / MemorySpace', ['Buffer 属于设备运行时的 Client。', 'memory_space 指明存储空间，device 提供设备归属。', 'Buffer 不只是一个裸地址。'], 'Buffer definition'),
             node('Future / 生命周期', ['句柄返回不保证数据已就绪。', 'Delete / donation 后不能继续按普通输入使用。', '删除前取得的就绪 Future 有其独立有效性约定。'], 'GetReadyFuture', 'Buffer delete'),
             node('ExternalReference / 所有权', ['外部引用可以持有底层存储并访问不透明指针。', '共享引用、释放所有权和复制是不同操作。', '跨框架读写需要满足同步约束。'], 'Buffer external ref', 'Buffer release'),
         ],
         producers=[
             node('BufferFromHostBuffer', ['输入：主机指针 + 类型 / shape / strides + 目标位置。', '遵守 HostBufferSemantics 和回调约定。', '输出：新的 PjRtBuffer，底层传输可异步完成。'], 'BufferFromHostBuffer'),
             node('PjRtLoadedExecutable::Execute', ['输入：每设备的输入 Buffer 列表 + 执行选项。', 'provider 为执行结果提供存储与返回句柄。', '输出：按设备、按结果组织的 Buffer 列表。'], 'PJRT Execute'),
             node('从已有 Buffer 派生', ['输入：已有 Buffer + 目标位置或新的解释方式。', 'CopyToMemorySpace / Bitcast 等接口创建新的 Buffer 句柄。', '输出仍是 Buffer；对应的存储与所有权变化在下方展开。'], 'CopyToMemorySpace', 'Buffer bitcast'),
         ],
         transforms=[
             node('CopyToMemorySpace', ['输入：Buffer + 目标 MemorySpace。', '按 provider 能力复制；目标已是当前空间时可报错。', '输出：目标存储中的新 Buffer；不是编译 lowering。'], 'CopyToMemorySpace'),
             node('Bitcast', ['输入：Buffer + 新 element_type / dims / layout。', '要求相同设备字节大小；只调整解释元数据，不复制数据。', '输出：新 Buffer，原 Buffer 的存储所有权被捐赠。'], 'Buffer bitcast'),
             node('DonateWithControlDependency', ['输入：Buffer + 额外依赖 Future。', '新句柄在数据和额外依赖均就绪后可用。', '输出：相同内容的新 Buffer；具体 provider 可不支持。'], 'Buffer dependency'),
             node('Delete / ReleaseDeviceMemoryOwnership', ['输入：当前 Buffer；可附等待选项。', '丢弃引用，或把存储所有权转交 ExternalReference。', '输出：原句柄失效；释放内存的时机受异步使用约束。'], 'Buffer delete', 'Buffer release'),
         ],
         after='新 Buffer / 原 Buffer 的新状态', after_rows=['复制、重解释和捐赠可以返回新句柄；Delete / Release 使原句柄失效。', '值、物理存储和句柄生命周期需分别理解。'],
         consumers=[
             consume('PjRtLoadedExecutable::Execute', ['已编译程序读取输入 Buffer 的参数列表。', '满足设备分配和可寻址性；按配置实施 donation。'], '结果 Buffer + Future', ['编译输入与执行数据在此阶段汇合。', '返回 Future 可报告对应设备的执行完成。'], 'PJRT Execute'),
             consume('ToLiteral / CopyRawToHost', ['读取 Buffer，将值或指定字节范围传给主机。', '不把设备句柄本身当作主机数组。'], 'Literal / 主机字节', ['异步读取有完成 Future。', '调用方需维持目标存储的有效期。'], 'Buffer to literal'),
             consume('GetReadyFuture', ['观察 Buffer 的计算、传输或错误状态。', '它是依赖接口，不是新的设备运算。'], '可等待状态', ['调用方按需等待。', '就绪不表示已把数据复制回主机。'], 'GetReadyFuture'),
             consume('AcquireExternalReference', ['对 Buffer 的底层设备存储取得外部引用。', '可供跨框架互操作，读取不透明设备指针。'], 'ExternalReference', ['引用保持存储存活；同步责任依约定承担。', '不是复制出一份 Buffer 数据。'], 'Buffer external ref'),
         ], foot='PJRT 编译接收 Module / 程序；Execute 接收 Buffer。输入 Buffer 不会被 lowering 成 HLO，捐赠语义不等于普通引用共享。'),

    dict(key='xla', path='xla/hlo-centered-hub.svg', title='XLA · 以 HLO 为中心', core='HLO / HloModule', core_ref='HloModule',
         summary='优化围绕有类型的计算图进行；各后端选择不同 pass，并在消费 HLO 时产生低层表示与执行计划。',
         center=['HloModule 持有入口和嵌套 HloComputation。', 'HloComputation 包含 HloInstruction 与根结果。',
                 'Instruction 以 operands / users 连接数据依赖。', 'Shape、Layout、sharding、metadata 与 config 共同约束程序。',
                 'HLO 是程序表示；设备 Buffer 是执行数据。'],
         definitions=[
             node('Module / Computation', ['入口 computation 定义调用边界和输出。', '嵌套计算承载调用、控制流或 fusion 的内部计算。', 'ModuleConfig 保存编译与布局等配置。'], 'HloModule', 'HloComputation'),
             node('Instruction / Opcode', ['操作码、operands、shape 与专属属性描述操作。', 'users 反向记录使用关系；root 指令给出结果。', '控制依赖与效果约束不能仅按纯数据 DAG 忽略。'], 'HloInstruction'),
             node('Shape / Layout / Sharding', ['shape 描述数组、元组及元素类型。', 'layout 和 sharding 表达存储 / 分片约束。', 'pass 的合法性取决于所处阶段与配置。'], 'HloModule', 'HloInstruction'),
             node('良构性与合法重写', ['HloVerifier 检查各计算中指令的不变量。', '部分验证对布局敏感；变换必须保持所需语义。', '图中变换不是所有平台统一的固定列表。'], 'HloVerifier'),
         ],
         producers=[
             node('MLIR → HLO 导入', ['输入：ModuleOp + MlirToHloConversionOptions。', '导出 HloProto，构造 ModuleConfig，再 CreateFromProto。', '输出：HloModule；StableHLO 与 HLO 是不同表示。'], 'ConvertMlirHloToHloModule'),
             node('HloModule::CreateFromProto', ['输入：HloModuleProto + config。', '恢复计算、指令及其引用关系。', '输出：HloModule；序列化描述不含执行 Buffer。'], 'HLO from proto'),
             node('HloModule / AddEntryComputation', ['输入：名称、config 和已构建的 HloComputation。', '通过模块构造和入口 / 嵌套计算的添加组织程序。', '输出：可继续校验和变换的 HLO 模块。'], 'HLO entry computation', 'HloComputation'),
         ],
         transforms=[
             node('HloPassPipeline', ['输入：HloModule + execution_threads / debug options。', 'RunImpl 按所配置的 pass 调度；支持原地与模块替换。', '输出：更新后的 HLO + changed / status。'], 'HloPassPipeline RunImpl'),
             node('AlgebraicSimplifier', ['输入：HLO + 代数简化选项。', '按操作语义和布局敏感性进行局部重写。', '输出：简化后的 HLO；并非任意代数恒等式都适用。'], 'AlgebraicSimplifier'),
             node('HloDCE', ['输入：HLO + 死参数 / collective 等处理选项。', '删除无用指令或计算；副作用和接口变化受约束。', '输出：裁剪后的 HLO + 是否改变。'], 'HloDCE'),
             node('Clone / ReplaceComputations', ['输入：已有 HloModule + 替换映射或 clone 配置。', '深拷贝计算，或重接指定 computation 的使用点。', '输出：新 HLO 对象或已更新的同一模块。'], 'HLO clone', 'HLO replace computations'),
         ],
         after='变换后的 HLO / HloModule', after_rows=['仍由 computation 与 instruction 构成；可能原地更新或返回新模块。', '融合、布局、分片和调度阶段由目标后端及配置决定。'],
         consumers=[
             consume('CpuCompiler::RunBackend', ['读取经过目标相关处理的 HLO。', '进入 CPU 编译和内存 / 执行规划。'], 'LLVM IR + CPU 可执行对象', ['IR emitter、机器码编译和 thunk 计划共同工作。', '也可能使用 CPU 库调用。'], 'CpuCompiler RunBackend'),
             consume('GpuCompiler::RunBackend', ['读取 HLO 和 GPU 目标信息。', '选择原生 emitter、Triton 或设备库等路径。'], '设备代码 + GPU 执行计划', ['生成 kernel 的路线和库调用并列。', '不是每条 HLO 对应一个 kernel。'], 'GpuCompiler RunBackend'),
             consume('HloVerifier', ['遍历 HLO 模块中的计算与指令。', '按 verifier 配置检查不变量。'], '验证状态 / 诊断', ['通过校验不证明设备已执行。', '验证也可以作为 pipeline 内的检查步骤。'], 'HloVerifier'),
             consume('ToString / ToProto', ['读取模块和打印 / 序列化选项。', '产生供诊断或传输的程序描述。'], 'HLO 文本 / HloModuleProto', ['文字和 proto 描述程序，而非实际数据 Buffer。', '后续可继续反序列化或分析。'], 'HLO text', 'HLO proto'),
         ], foot='TPU provider 同样接受编译输入，但本地开源源码不能展开其全部内部优化。HLO 到后端是一对多路径，不能与硬件指令一一对应。'),
]

DIAGRAMS += [
    dict(key='cpu', path='xla/cpu-llvm-ir-centered-hub.svg', title='XLA CPU 后端 · 以 LLVM IR 为中心',
         core='CPU LLVM IR / llvm::Module', core_ref='LLVM Module',
         summary='在 CPU 后端中，HLO 产生 LLVM IR；LLVM 优化与代码生成消费该表示，执行由机器码和 thunk 计划共同实现。',
         center=['llvm::Module 持有 Function、全局值和元数据。', 'Function 包含 BasicBlock；块中的 Instruction 构成控制流与计算。',
                 'TargetTriple / DataLayout 绑定目标和数据表示。', 'kernel 函数遵循运行时调用约定，参数关联分配好的存储。',
                 'LLVM IR 与 ThunkSequence 是协作产物，不是同一种对象。'],
         definitions=[
             node('Module / Function', ['Module 管理函数、全局变量和目标属性。', 'Function 定义参数、返回类型和函数体。', 'LLVMContext 管理相关类型和常量等对象。'], 'LLVM Module', 'LLVM Function'),
             node('BasicBlock / Instruction', ['基本块持有指令序列，终结指令描述控制流。', 'Value / Use 关系连接操作数与计算结果。', '这一层使用 LLVM 的类型、指令与语义约束。'], 'LLVM BasicBlock', 'LLVM Instruction'),
             node('TargetTriple / DataLayout', ['CompileCpuExecutable 根据 TargetMachine 设置目标。', 'DataLayout 约束大小、对齐及指针等低层表示。', '目标 CPU 特性影响可用指令和优化决策。'], 'CpuCompiler CompileCpuExecutable'),
             node('Kernel 与执行计划', ['IrEmitter2 为计算构造 LLVM host kernels。', 'ThunkEmitter 组织 kernel 和库调用的运行顺序。', 'BufferAssignment 等运行时约定不全在 IR 中。'], 'CPU IrEmitter2', 'CPU ThunkEmitter'),
         ],
         producers=[
             node('CompileCpuExecutable 创建 Module', ['输入：HloModule + IrCompiler / target 配置。', '创建 LLVMContext、Module，设置 triple 与 data layout。', '输出：可填充函数与全局值的目标 Module。'], 'CpuCompiler CompileCpuExecutable'),
             node('IrEmitter2 生成 host kernels', ['输入：HLO 计算 + LLVM Module + 嵌套 emitter。', '为融合与元素级计算等组织函数、循环与指令。', '输出：Module 内的 LLVM 函数与 kernel 信息。'], 'CPU IrEmitter2'),
             node('ThunkEmitter 的额外 kernels', ['输入：入口 computation + buffer / 目标信息。', 'EmitEntryComputation 组织工作并收集独立 kernel 模块。', '输出：LLVM kernel 模块，同时产生 ThunkSequence。'], 'CPU ThunkEmitter'),
         ],
         transforms=[
             node('IrCompiler::RunIrPasses', ['输入：llvm::Module + TargetMachine / 优化选项。', 'PassBuilder 配置分析与优化 pipeline；pm.run 修改 Module。', '输出：优化后的 LLVM IR；O0 与其他级别分支不同。'], 'CPU IR passes'),
             node('RemoveUnusedSymbols', ['输入：待编译 LLVM 模块片段。', '删除不需要的符号，整理编译单元。', '输出：符号范围收敛后的 Module。'], 'CPU unused symbols'),
             node('ExtractKernelsFromModule', ['输入：LLVM Module + 需单独处理的 kernels。', '部分 kernel 因后端选项不同而独立编译。', '输出：拆出的 kernel Module 和调整后的主体。'], 'CPU extra kernels'),
             node('llvm::SplitModule', ['输入：LLVM Module + 并行编译分片数。', '按条件拆分模块，并交给多个编译单元。', '输出：多个 LLVM Module；这是编译分片而非数组分片。'], 'CPU split module'),
         ],
         after='优化 / 拆分后的 LLVM Module', after_rows=['仍是 LLVM IR；可以是同一模块的更新，或多个独立编译单元。', '编译线程分工不会把它变成 IFRT Array 或 PJRT Buffer。'],
         consumers=[
             consume('IrCompiler::EmitMachineCode', ['读取 LLVM Module + TargetMachine。', 'LLVM 目标代码生成 pass 输出目标对象字节。'], 'CPU 目标对象 / 机器码', ['机器指令受目标 ISA、ABI 与特性约束。', 'ASM 是代码的汇编表示，不是强制独立中间站。'], 'CPU machine code'),
             consume('AddModule / Compile', ['把 LLVM Module 交给 JIT 或 AOT 模块编译器。', '收集并解析编译后的 kernel / comparator 符号。'], '编译符号 + CpuExecutable', ['代码与 ThunkSequence、常量、内存计划共同组成可执行对象。', 'CPU 库调用可直接进入执行计划。'], 'CPU AddModule'),
             consume('VerifyLlvmModule', ['LLVM verifier 读取 Module。', '失败时报告非法低层程序。'], '验证状态', ['这是 IR 合法性证据。', '不会执行 CPU kernel，也不证明数值或性能。'], 'CPU IR verification'),
             consume('编译钩子 / IR dump', ['编译前后钩子观察 LLVM Module。', '可输出优化前后的 IR 文本。'], '诊断文本 / 观察结果', ['用于定位代码生成与优化。', '打印出来的 IR 不等于执行 trace。'], 'CPU IR compiler'),
         ], foot='所属组件：XLA CPU 后端；LLVM 是使用的编译基础设施。实际程序可能混合生成 kernel 与 oneDNN / Eigen 等库调用。'),

    dict(key='gpu', path='xla/gpu-ir-centered-hub.svg', title='XLA GPU 后端 · 以 LLVM IR / Triton IR 为中心',
         core='GPU LLVM IR / Triton IR', core_ref='GPU module emission',
         summary='保留多条代码生成路径：原生 emitter 与 Triton 路径可产生代码，库调用另行纳入执行计划。',
         center=['LLVM 路：llvm::Module / Function / 指令 + GPU 目标属性。', 'Triton 路：MLIR Module + Triton / XLA 等方言操作。',
                 'Triton 编译可以生成 LLVM IR；并非所有 GPU 代码先经过 Triton。', '线程块、warp、共享内存等配置参与代码生成。',
                 'IR 与 kernel 二进制、launch 配置、ThunkSequence 分别表示不同信息。'],
         definitions=[
             node('LLVM Module / GPU 目标', ['函数、指令与全局值由 LLVM IR 表达。', 'GPU target triple、data layout 与 kernel 标记参与代码生成。', '目标设备特性影响最终机器指令。'], 'LLVM Module', 'GPU module emission'),
             node('Triton 的 MLIR 容器', ['TritonKernelSource 持有 MLIR Module。', '初始模块可含 XLA / StableHLO 等操作，逐步转换。', '此处的 Triton 路径是 XLA 的一条代码生成分支。'], 'Triton fusion', 'Triton XLA pipeline'),
             node('Tile / Warp / Shared memory', ['BlockLevelParameters 与设备信息约束 kernel 构造。', 'num_warps / num_ctas / num_stages 参与目标 pipeline。', '配置不是一条 HLO 到一条指令的映射表。'], 'Triton fusion', 'Triton target pipeline'),
             node('代码与库调用的边界', ['CompileModuleToLlvmIr 同时组织可执行 thunk 结构。', 'cuBLAS / cuDNN 等库路径无需现场生成每个操作的 kernel。', 'kernel 代码和库调用共同构成 GPU 可执行计划。'], 'GPU module emission', 'GpuCompiler RunBackend'),
         ],
         producers=[
             node('CompileModuleToLlvmIr', ['输入：HLO + target / buffer / alias / 设备信息。', 'IrEmitterContext 与 ThunkEmitter 组织代码和工作计划。', '输出：LLVM 代码生成产物、常量与 thunk 结果。'], 'GPU module emission'),
             node('TritonFusion / CreateTritonModule', ['输入：已选为 Triton fusion 的 HLO + 块级参数。', 'GenerateTritonKernelAndWrapper 创建 TritonKernelSource。', '输出：MLIR 模块，再交给 Triton 到 LLVM 编译接口。'], 'Triton fusion'),
         ],
         transforms=[
             node('CreateTritonXlaPipeline', ['输入：含 XLA / 高层操作的 Triton MLIR Module。', '组合 shape、transpose、索引及高层操作合法化规则。', '输出：更接近 Triton 目标编译输入的 MLIR。'], 'Triton XLA pipeline'),
             node('CreateTritonPipeline', ['输入：Triton IR + GPU capability / warp / CTA 配置。', '按 CUDA 或 ROCm 目标选用不同 pass pipeline。', '输出：对应目标路径中的低层 MLIR 表示。'], 'Triton target pipeline'),
             node('CompileTritonToLlvm', ['输入：TritonKernelSource + HLO / target / 块级上下文。', '调度 CompileTritonToLLVM，可在线程池异步执行。', '输出：TritonWrapperResult，供 LLVM kernel 包装与后续编译。'], 'Triton to LLVM', 'Triton fusion'),
             node('LinkAndOptimizeModule', ['输入：GPU LLVM Module + TargetMachine / 选项。', '按需要链接设备 bitcode，并运行 LLVM 优化。', '输出：适合目标代码生成的 LLVM Module。'], 'GPU link and optimize'),
         ],
         after='转换 / 优化后的 GPU IR', after_rows=['Triton 分支可到 LLVM IR；原生 LLVM 分支直接继续目标编译。', '上方四框表示可选路径与阶段，不能逐框串成所有 GPU 程序的统一流程。'],
         consumers=[
             consume('NVPTXCompiler::CompileTargetBinary', ['NVIDIA 路读取 LLVM Module 与设备 capability。', '生成 PTX，再由 compilation provider 编译或链接。'], 'PTX / cubin 二进制', ['PTX 是虚拟 ISA 的表示；cubin 承载目标 GPU 代码。', '目标汇编与实际 ISA 契约不是强制的 ASM → ISA 转换。'], 'NVPTX target binary'),
             consume('GpuCompiler::CompileSingleModule', ['读取单个 LLVM Module + HLO 配置 / 设备描述。', '检查 IR 并委派给目标二进制编译接口。'], 'BackendCompileResult', ['返回目标二进制与编译统计等信息。', '上层把结果与 thunk、常量和存储计划组合。'], 'GPU single module'),
             consume('LLVM verifier', ['在目标编译前检查 LLVM Module。', '非法 LLVM IR 导致编译诊断。'], '验证状态 / 错误', ['只证明 IR 满足检查条件。', '不等于 GPU 已运行或性能已验证。'], 'GPU LLVM verification'),
             consume('IR dump / 编译观察', ['读取编译模块和优化阶段。', 'CompileSingleModule 按配置输出 LLVM IR。'], 'IR 文本与编译信息', ['用于检查生成的代码表示。', '不能从静态 IR 单独推断真实设备重叠或吞吐。'], 'GPU single module'),
         ], foot='目标二进制一栏以本地 NVIDIA 编译实现为具体证据；ROCm 具有另一路目标后端，不应把所有 GPU 都标成 PTX / cubin。'),

    dict(key='tpu', path='libtpu/llo-boundary-centered-hub.svg', title='libtpu / TPU 后端 · 以 LLO 为研究中心的证据边界',
         core='LLO · 内部结构待确认', core_ref='TPU plugin', uncertain=True,
         summary='沿用四视角构图，但开源接口只能证明 TPU 插件的输入输出；不能据此补造 LLO 的类、字段、pass 或机器指令。',
         center=['研究对象：用户指定的 TPU 后端内部表示 LLO。', '当前固定开源源码未提供可核验的 LLO 结构和完整变换链。',
                 '可见边界：libtpu 插件接收程序，返回已加载可执行对象。', 'LLO 在私有后端中的阶段、格式与消费者仍待对应版本证据。',
                 '外围引用只证明公开接口，不证明这些接口直接操作 LLO。'],
         definitions=[
             node('LLO 的结构与语义', ['类、字段、操作集合、类型与效果约束：尚未确认。', '不能用 HLO、Mosaic MLIR 或 TPU 程序句柄代替定义。', '需要对应 libtpu 版本的结构说明或真实产物。'], tag='证据缺口', uncertain=True),
             node('公开程序输入', ['PJRT_Program 提供 code、code_size 与 format。', '公开格式包括 MLIR / HLO 描述，不给出 LLO schema。', '这属于插件输入契约，不能推断后端全部 IR。'], 'PJRT Program'),
             node('不透明程序句柄', ['公开头文件中 XLA_TpuProgram 仅有前向声明。', '句柄可见不意味着能读取其内部程序结构。', '此 C API 声明也不能等同于当前 PJRT 插件完整实现。'], 'TPU opaque program'),
             node('Mosaic 与 LLO 的区别', ['Pallas 分支可生成 Mosaic TPU MLIR。', '它被序列化进外层 custom_call，随后交给后端。', 'Mosaic TPU MLIR 不是 LLO，也不是 TPU 机器码。'], 'lower_module_to_custom_call'),
         ],
         producers=[
             node('可见入口：TPU 插件加载', ['make_tpu_client 加载并初始化 libtpu.so 的 PJRT 插件。', '获得设备 Client，供上层提交程序。', '此入口不公开 LLO 的构造函数。'], 'TPU plugin', tag='公开边界，不是 LLO 生成位置'),
             node('可见入口：PJRT 编译 ABI', ['输入：PJRT_Program + serialized CompileOptions。', '输出契约：PJRT_LoadedExecutable。', '程序到 LLO 的内部构造者和确切输入尚未定位。'], 'PJRT compile ABI', tag='内部产生路径待确认', uncertain=True),
         ],
         transforms=[
             node('变换由谁执行？', ['尚未找到可核验的 LLO pass 类或函数。', '公开编译入口只显示请求与结果。', '不得据此列出假定的内部 pass 名称。'], 'PJRT compile ABI', tag='待取证的问题', uncertain=True),
             node('变换的输入输出是什么？', ['LLO 文件格式、schema、版本和中间阶段未确认。', '需要能与当前 libtpu 构建对应的真实产物。', '不能把任意 .pb 或 MLIR 文本默认标为 LLO。'], tag='待取证的问题', uncertain=True),
             node('哪些规则保持语义？', ['LLO verifier、效果和合法重写约束尚未确认。', '需内部接口或版本化文档支持。', '不能从 HLO verifier 自动外推。'], tag='待取证的问题', uncertain=True),
             node('阶段如何衔接？', ['与布局、调度、内存规划、指令选择的关系待确认。', '这些是研究问题，不是本图声称存在的固定阶段。', '公开 API 不足以给出完整顺序。'], tag='待取证的问题', uncertain=True),
         ],
         after='LLO 的变换产物 · 待取证', after_rows=['这里没有声称已获得新的 LLO 对象。', '虚线只连接研究问题与证据边界，不表示已证实的调用链。'],
         consumers=[
             consume('内部 LLO 消费者', ['具体类、函数、调用位置与输出格式尚未确认。', '需要后端内部或版本对应的编译产物证据。'], '私有后端结果 · 未确认', ['不能补造 LLO → TPU 指令的具体接口。', '不能将 Mosaic 模块冒充最终机器码。'], uncertain=True),
             consume('公开编译结果', ['PJRT_Client_Compile 返回 LoadedExecutable 句柄。', '只证明编译的外部输出契约。'], 'PJRT_LoadedExecutable', ['句柄封装可加载 / 执行程序的行为。', '不对外暴露 LLO 的完整语义。'], 'PJRT compile ABI', uncertain=True),
             consume('公开执行与序列化', ['Execute 接收 Buffer；Serialize 处理可执行对象。', '这些不是已知的 LLO 读取接口。'], '输出 Buffer / 可执行字节', ['执行事件描述完成；序列化兼容性由实现决定。', '不能由这些返回值反推内部 LLO schema。'], 'PJRT execute ABI', 'PJRT serialize ABI', uncertain=True),
         ], foot='本图是可审查的证据边界图；未运行 TPU、未提取 LLO、未确认 libtpu 私有 pass。虚线表示待确认关系，绝不表示确定的编译转换。'),

    dict(key='runtime', path='stream-executor/submission-centered-hub.svg', title='StreamExecutor / CUDA · 以执行提交为中心',
         core='执行提交 · 请求与异步工作', core_ref='Stream launch', action=True,
         summary='以公开 CUDA 路径展开设备运行时：程序与数据在提交处汇合，依赖、队列状态、完成事件围绕这一动作协作。',
         center=['研究对象是执行提交动作；不是一个名为 Submit 的统一 IR 类。', 'Stream 表达按序异步工作，CUDA 实现持有 CUstream。',
                 'kernel 句柄 + launch 维度 + 参数地址 + shared memory 描述启动。', '依赖通过 stream / event 建立；状态与完成需要单独观察。',
                 'CPU 使用自身执行机制，TPU 插件内部不由 CUDA 源码证明。'],
         definitions=[
             node('Stream / CUstream', ['同一 stream 的操作按序异步执行。', 'Stream 接口对接具体平台句柄。', '不同 stream 的依赖需要事件或等待关系表达。'], 'Stream'),
             node('LaunchKernel 的参数契约', ['ThreadDim / BlockDim / 可选 ClusterDim 描述启动维度。', 'function、args、shmem_bytes 与设备配置构成调用参数。', '这些参数不是 LLVM IR，也不是源语言数组对象。'], 'Stream launch'),
             node('DeviceAddress / BufferAllocations', ['执行计划从 BufferAllocations 解析设备存储地址。', 'kernel 参数和加载的函数共同确定一次具体工作。', '存储必须覆盖异步工作使用它的生命周期。'], 'Kernel arguments'),
             node('Event / Status / 完成', ['提交返回的 Status 与工作完成状态是不同观察点。', 'RecordEvent / WaitFor 建立设备侧依赖。', 'BlockHostUntilDone 使主机等待 stream 工作完成。'], 'Stream record', 'Stream synchronize'),
         ],
         producers=[
             node('KernelThunk::ExecuteOnStream', ['输入：已加载 kernel、BufferAllocations 与执行 stream。', 'GetKernelAndArgs 解析实际参数；按 launch 配置启动。', '输出：提交到 stream 的工作及提交状态。'], 'Kernel submission', 'Kernel arguments'),
             node('CommandBufferThunk::ExecuteOnStream', ['输入：记录的命令计划 + 当前设备地址和 stream。', '按需更新命令，调用 command_buffer->Submit。', '输出：一次命令缓冲区提交；可能包含多个工作项。'], 'Command buffer submission'),
             node('Stream::Memcpy / 平台实现', ['输入：主机或设备地址、字节数与 stream。', 'CUDA Memcpy 实现把拷贝作为异步工作提交。', '输出：传输工作；拷贝与 kernel 都需要依赖和完成管理。'], 'Stream', 'CUDA memcpy', tag='另一类设备工作：传输'),
         ],
         transforms=[
             node('RecordEvent / WaitFor', ['输入：已有异步工作 + event / 另一条 stream。', '记录完成点并增加后续工作的等待关系。', '输出：受新依赖约束的提交顺序；不改写编译 IR。'], 'Stream record', 'Stream wait'),
             node('CUDA graph capture', ['输入：捕获区间中的 stream 提交。', 'CaptureHandle 管理开始、结束与恢复 stream。', '输出：可记录为图的工作；只有相应路径使用捕获。'], 'CUDA graph capture'),
             node('更新已记录命令', ['输入：CommandBuffer + 当前 BufferAllocations。', '检查地址变化，按需重录或更新命令后再提交。', '输出：绑定当前地址的可提交工作计划。'], 'Command buffer submission'),
             node('完成状态推进', ['输入：已排队工作及设备执行进展。', '事件和同步接口报告成功或失败，主机可按需等待。', '输出：已完成 / 错误状态；不是创建另一种 IR。'], 'CUDA synchronize', 'Stream'),
         ],
         after='更新的提交关系 / 工作状态', after_rows=['变化的是依赖、记录计划、地址绑定或完成状态。', '各机制按需使用；不会把运行时数据变成 HLO 或 LLVM IR。'],
         consumers=[
             consume('CudaStream::LaunchKernel', ['读取具体 launch 参数与 stream。', 'LaunchCudaKernel 调用 cuLaunchKernel / cuLaunchKernelEx。'], '驱动队列中的 kernel 工作', ['驱动返回提交状态，硬件随后执行目标设备程序。', '本地源码能确认提交调用，不能展示驱动私有内部。'], 'CUDA launch', 'CUDA driver launch'),
             consume('CUDA 异步 memcpy', ['读取传输请求的源 / 目标地址、字节数和 stream。', '调用 cuMemcpyHtoDAsync 等驱动接口。'], '排队传输 / 存储变化', ['提交状态与传输完成分别观察。', '调用方必须维持相关存储的有效期。'], 'CUDA memcpy'),
             consume('后续 stream 工作', ['读取 event 或另一条 stream 的完成依赖。', 'WaitFor 约束后续工作开始的时机。'], '满足依赖后的继续执行', ['建立正确性所需的先后关系。', '是否实际并行或重叠需要真实设备测量。'], 'Stream wait'),
             consume('BlockHostUntilDone', ['主机消费 stream 的异步完成状态。', 'CUDA 实现同步并检查错误。'], '主机可观察的成功 / 失败', ['等待完成与回传数组是不同动作。', '从 Buffer 到 Array 的包装由上层运行时完成。'], 'CUDA synchronize'),
         ], foot='StreamExecutor / CUDA 为具体证据；CPU / TPU 各有实现。设备机器码按 ISA 契约执行，ASM 只是表示，不存在强制 ASM → ISA 转换阶段。'),
]
