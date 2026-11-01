use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub const PROGRAM_FORMAT: &str = "prog-ir/0.1";
pub const ALGORITHM_FORMAT: &str = "algo-ir/0.1";
pub const FLOW_FORMAT: &str = "uniform-flow-ir/1.0";
pub const ARCH_FORMAT: &str = "arch-ir/0.1";
pub const KERNEL_DAG_FORMAT: &str = "gtsim-kernel-dag/0.1";
pub const EXECUTION_PROGRAM_FORMAT: &str = "gtsim-execution-program/0.1";
pub const HARDWARE_PROFILE_FORMAT: &str = "hardware-profile/0.1";

pub type Attributes = BTreeMap<String, Value>;

/// A dimension is either a compile-time constant or a runtime symbol.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Dimension {
    Static(u64),
    Runtime(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorDecl {
    pub id: String,
    pub name: String,
    pub shape: Vec<Dimension>,
    pub dtype: String,
    pub storage: String,
    #[serde(default)]
    pub external: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeParam {
    pub name: String,
    pub dtype: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KernelInvocation {
    pub id: String,
    pub callee: String,
    #[serde(default)]
    pub inputs: Vec<String>,
    #[serde(default)]
    pub outputs: Vec<String>,
    #[serde(default)]
    pub attrs: Attributes,
    #[serde(default)]
    pub after: Vec<String>,
    pub completion_event: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProgramCommand {
    Launch { invocation: KernelInvocation },
    RecordEvent { event: String },
    WaitEvent { event: String },
    Synchronize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stream {
    pub name: String,
    pub ordered: bool,
    pub commands: Vec<ProgramCommand>,
}

/// IR-01: program-level invocation orchestration and global tensor identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub format: String,
    pub name: String,
    #[serde(default)]
    pub params: Vec<RuntimeParam>,
    #[serde(default)]
    pub tensors: Vec<TensorDecl>,
    pub streams: Vec<Stream>,
    #[serde(default)]
    pub returns: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexDomain {
    pub name: String,
    pub extent: Dimension,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperandContract {
    pub name: String,
    #[serde(default)]
    pub axes: Vec<String>,
    pub dtype: String,
    #[serde(default)]
    pub output: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MathOperation {
    pub id: String,
    pub kind: String,
    #[serde(default)]
    pub inputs: Vec<String>,
    pub output: String,
    #[serde(default)]
    pub reduction_axes: Vec<String>,
    #[serde(default)]
    pub output_axes: Vec<String>,
    #[serde(default)]
    pub dtype: String,
    #[serde(default)]
    pub constants: Attributes,
    #[serde(default)]
    pub parameter: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlgorithmStage {
    pub name: String,
    #[serde(default)]
    pub reads: Vec<String>,
    #[serde(default)]
    pub produces: Vec<String>,
    #[serde(default)]
    pub after: Vec<String>,
}

/// IR-02: mathematical semantics, index domains, stages and known algorithm family.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AlgorithmTemplate {
    pub format: String,
    pub name: String,
    #[serde(default)]
    pub domains: Vec<IndexDomain>,
    #[serde(default)]
    pub operands: Vec<OperandContract>,
    #[serde(default)]
    pub operations: Vec<MathOperation>,
    #[serde(default)]
    pub stages: Vec<AlgorithmStage>,
    pub family: Vec<String>,
    #[serde(default)]
    pub invariants: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,
    pub min: u64,
    #[serde(default)]
    pub max: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorRegion {
    pub tensor: String,
    #[serde(default)]
    pub offsets: Vec<Dimension>,
    #[serde(default)]
    pub shape: Vec<Dimension>,
    #[serde(default)]
    pub mask: Option<String>,
    #[serde(default)]
    pub layout: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Executor {
    pub id: String,
    pub role: String,
    pub count: u32,
    pub organization: String,
    pub progress: String,
    pub lifetime: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FlowBuffer {
    pub id: String,
    pub slots: u32,
    pub access: String,
    pub shape: Vec<Dimension>,
    pub dtype: String,
    pub memory_class: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DependencyCondition {
    Issue,
    Complete,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dependency {
    pub operation: String,
    pub condition: DependencyCondition,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotUse {
    pub buffer: String,
    pub slot: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    #[serde(default)]
    pub bytes: Option<u64>,
    #[serde(default)]
    pub flops: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoopRange {
    pub variable: String,
    pub start: u64,
    /// Exclusive upper bound. Loops are retained until IR-04 -> IR-05.
    pub end: u64,
    pub step: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FlowOperation {
    pub id: String,
    pub kind: String,
    pub capability: String,
    pub executor: String,
    #[serde(default)]
    pub operands: Vec<String>,
    #[serde(default)]
    pub reads: Vec<TensorRegion>,
    #[serde(default)]
    pub writes: Vec<TensorRegion>,
    #[serde(default)]
    pub dependencies: Vec<Dependency>,
    #[serde(default)]
    pub loops: Vec<LoopRange>,
    #[serde(default)]
    pub slots: Vec<SlotUse>,
    #[serde(default)]
    pub work: Work,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecuteBinding {
    pub executor: String,
    pub capability: String,
    pub engine: String,
    /// Instruction selection is explicit in Mapping, never guessed by the lowerer.
    pub opcode: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreBinding {
    pub buffer: String,
    pub memory: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorTierBinding {
    pub tensor: String,
    pub memory: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteBinding {
    pub capability: String,
    pub from: String,
    pub to: String,
    pub via: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CtaPlacement {
    pub count: u32,
    pub warps: u32,
    pub threads: u32,
    pub shared_memory_bytes: u64,
    pub registers_per_thread: u32,
}

/// Mapping is an adjunct embedded in IR-03, not a sixth IR layer or mapir file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceMapping {
    pub flow: String,
    pub hardware: String,
    #[serde(default)]
    pub executes: Vec<ExecuteBinding>,
    #[serde(default)]
    pub stores: Vec<StoreBinding>,
    #[serde(default)]
    pub tensors: Vec<TensorTierBinding>,
    #[serde(default)]
    pub routes: Vec<RouteBinding>,
    pub cta: CtaPlacement,
}

/// IR-03: scheduled logical dataflow and its physical ResourceMapping adjunct.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FlowKernel {
    pub format: String,
    pub name: String,
    pub invocation_id: String,
    pub template_id: String,
    pub schedule_id: String,
    #[serde(default)]
    pub symbols: Vec<Symbol>,
    #[serde(default)]
    pub tensors: Vec<TensorDecl>,
    #[serde(default)]
    pub executors: Vec<Executor>,
    #[serde(default)]
    pub buffers: Vec<FlowBuffer>,
    #[serde(default)]
    pub operations: Vec<FlowOperation>,
    pub mapping: ResourceMapping,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Engine {
    pub name: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
}

/// Hardware description is an external target profile, not one of the five IRs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub format: String,
    pub id: String,
    pub target: String,
    pub max_shared_memory_bytes_per_cta: u64,
    pub max_registers_per_thread: u32,
    pub max_warps_per_cta: u32,
    #[serde(default)]
    pub memory_spaces: Vec<String>,
    #[serde(default)]
    pub engines: Vec<Engine>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeCta {
    pub id: u32,
    pub threads: u32,
    pub warps: u32,
    pub shared_memory_bytes: u64,
    pub registers_per_thread: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeBuffer {
    pub id: String,
    pub memory: String,
    pub slots: u32,
    pub bytes_per_slot: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeBarrier {
    pub id: String,
    pub kind: String,
    pub phases: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperation {
    pub id: String,
    pub opcode: String,
    pub executor: String,
    #[serde(default)]
    pub reads: Vec<TensorRegion>,
    #[serde(default)]
    pub writes: Vec<TensorRegion>,
    #[serde(default)]
    pub dependencies: Vec<Dependency>,
    #[serde(default)]
    pub loops: Vec<LoopRange>,
    #[serde(default)]
    pub work: Work,
}

/// IR-04: target-specific operations, layouts, synchronization and resources.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchKernel {
    pub format: String,
    pub name: String,
    pub invocation_id: String,
    pub template_id: String,
    pub schedule_id: String,
    pub target: String,
    #[serde(default)]
    pub tensors: Vec<TensorDecl>,
    #[serde(default)]
    pub ctas: Vec<NativeCta>,
    #[serde(default)]
    pub buffers: Vec<NativeBuffer>,
    #[serde(default)]
    pub barriers: Vec<NativeBarrier>,
    #[serde(default)]
    pub operations: Vec<NativeOperation>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DagNode {
    pub id: String,
    pub cta_id: u32,
    #[serde(default)]
    pub warp_id: Option<u32>,
    pub opcode: String,
    #[serde(default)]
    pub iteration: BTreeMap<String, u64>,
    #[serde(default)]
    pub reads: Vec<TensorRegion>,
    #[serde(default)]
    pub writes: Vec<TensorRegion>,
    #[serde(default)]
    pub after_issue: Vec<String>,
    #[serde(default)]
    pub after_complete: Vec<String>,
    #[serde(default)]
    pub work: Work,
}

/// IR-05 kernel DAG: a per-kernel graph, not a global program graph.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelDag {
    pub format: String,
    pub name: String,
    pub invocation_id: String,
    pub template_id: String,
    pub target: String,
    #[serde(default)]
    pub tensors: Vec<TensorDecl>,
    #[serde(default)]
    pub ctas: Vec<NativeCta>,
    #[serde(default)]
    pub nodes: Vec<DagNode>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorBinding {
    pub kernel_tensor_id: String,
    pub program_tensor_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionLaunch {
    pub id: String,
    pub invocation_id: String,
    pub kernel_dag: String,
    #[serde(default)]
    pub bindings: Vec<TensorBinding>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecutionCommand {
    Launch {
        invocation_id: String,
        #[serde(default)]
        after: Vec<String>,
        completion_event: String,
    },
    RecordEvent { event: String },
    WaitEvent { event: String },
    Synchronize,
}

/// IR-05 program manifest: one ordered stream and cross-kernel command semantics.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionProgram {
    pub format: String,
    pub name: String,
    pub streams: Vec<String>,
    #[serde(default)]
    pub tensors: Vec<TensorDecl>,
    #[serde(default)]
    pub launches: Vec<ExecutionLaunch>,
    #[serde(default)]
    pub commands: Vec<ExecutionCommand>,
    pub launch_overhead_cycles: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ir05Document {
    Kernel(KernelDag),
    Program(ExecutionProgram),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoweringProvenance {
    pub source_ir_hash: String,
    pub target_ir_hash: String,
    #[serde(default)]
    pub template_id: Option<String>,
    #[serde(default)]
    pub schedule_id: Option<String>,
    #[serde(default)]
    pub hardware_profile_id: Option<String>,
    #[serde(default)]
    pub tensor_mapping: BTreeMap<String, String>,
    #[serde(default)]
    pub invocation_mapping: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lowered<T> {
    pub ir: T,
    pub provenance: LoweringProvenance,
}
