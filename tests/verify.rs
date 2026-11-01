use llm_ir_lowerer::ir::*;
use llm_ir_lowerer::verify::*;

fn sample_flow() -> FlowKernel {
    FlowKernel {
        format: FLOW_FORMAT.into(),
        name: "flow0".into(),
        invocation_id: "gemm0".into(),
        template_id: "gemm".into(),
        schedule_id: "baseline".into(),
        symbols: vec![],
        tensors: vec![TensorDecl {
            id: "A".into(),
            name: "A".into(),
            shape: vec![Dimension::Static(128), Dimension::Static(64)],
            dtype: "bf16".into(),
            storage: "HBM".into(),
            external: true,
        }],
        executors: vec![Executor {
            id: "producer".into(),
            role: "producer".into(),
            count: 1,
            organization: "independent".into(),
            progress: "ordered".into(),
            lifetime: "per_instance".into(),
        }],
        buffers: vec![],
        operations: vec![FlowOperation {
            id: "load".into(),
            kind: "load_async".into(),
            capability: "bulk_copy".into(),
            executor: "producer".into(),
            operands: vec![],
            reads: vec![TensorRegion {
                tensor: "A".into(),
                offsets: vec![Dimension::Static(0), Dimension::Static(0)],
                shape: vec![Dimension::Static(128), Dimension::Static(64)],
                mask: None,
                layout: Some("row_major".into()),
            }],
            writes: vec![],
            dependencies: vec![],
            loops: vec![LoopRange {
                variable: "k".into(),
                start: 0,
                end: 2,
                step: 1,
            }],
            slots: vec![],
            work: Work { bytes: Some(16384), flops: None },
        }],
        mapping: ResourceMapping {
            flow: "flow0".into(),
            hardware: "h100".into(),
            executes: vec![],
            stores: vec![],
            tensors: vec![],
            routes: vec![],
            cta: CtaPlacement {
                count: 1,
                warps: 4,
                threads: 128,
                shared_memory_bytes: 16384,
                registers_per_thread: 32,
            },
        },
    }
}

#[test]
fn flow_verifier_accepts_a_well_formed_structured_loop() {
    assert!(verify_flow(&sample_flow()).is_ok());
}

#[test]
fn flow_verifier_rejects_a_dependency_cycle() {
    let mut flow = sample_flow();
    flow.operations[0].dependencies.push(Dependency {
        operation: "load".into(),
        condition: DependencyCondition::Complete,
    });
    assert!(verify_flow(&flow).unwrap_err().contains("self dependency"));
}

#[test]
fn flow_verifier_rejects_unknown_tensor_regions() {
    let mut flow = sample_flow();
    flow.operations[0].reads[0].tensor = "missing".into();
    assert!(verify_flow(&flow).unwrap_err().contains("unknown tensor"));
}
