use llm_ir_lowerer::codec::{decode, decode_ir05, encode_pretty};
use llm_ir_lowerer::ir::*;

#[test]
fn program_round_trips_with_canonical_format() {
    let program = Program {
        format: PROGRAM_FORMAT.into(),
        name: "decode_step".into(),
        params: vec![RuntimeParam {
            name: "kv_lengths".into(),
            dtype: "vector<index>".into(),
        }],
        tensors: vec![TensorDecl {
            id: "t0".into(),
            name: "hidden".into(),
            shape: vec![Dimension::Runtime("B".into()), Dimension::Static(2048)],
            dtype: "bf16".into(),
            storage: "HBM".into(),
            external: false,
        }],
        streams: vec![Stream {
            name: "compute".into(),
            ordered: true,
            commands: vec![ProgramCommand::Launch {
                invocation: KernelInvocation {
                    id: "norm0".into(),
                    callee: "rmsnorm".into(),
                    inputs: vec!["t0".into()],
                    outputs: vec![],
                    attrs: Default::default(),
                    after: vec![],
                    completion_event: "e0".into(),
                },
            }],
        }],
        returns: vec![],
    };

    let encoded = encode_pretty(&program).unwrap();
    let decoded: Program = decode(&encoded).unwrap();
    assert_eq!(decoded, program);
    assert!(encoded.contains("\"format\": \"prog-ir/0.1\""));
}

#[test]
fn ir05_uses_distinct_program_and_kernel_discriminators() {
    let kernel = KernelDag {
        format: KERNEL_DAG_FORMAT.into(),
        name: "gemm_dag".into(),
        invocation_id: "gemm0".into(),
        template_id: "gemm".into(),
        target: "sm90".into(),
        tensors: vec![],
        ctas: vec![],
        nodes: vec![],
    };
    let text = encode_pretty(&kernel).unwrap();
    assert!(matches!(decode_ir05(&text).unwrap(), Ir05Document::Kernel(_)));

    let program = ExecutionProgram {
        format: EXECUTION_PROGRAM_FORMAT.into(),
        name: "decode_step".into(),
        streams: vec!["compute".into()],
        tensors: vec![],
        launches: vec![],
        commands: vec![],
        launch_overhead_cycles: 0,
    };
    let text = encode_pretty(&program).unwrap();
    assert!(matches!(decode_ir05(&text).unwrap(), Ir05Document::Program(_)));
}

#[test]
fn ir05_rejects_unknown_format() {
    let err = decode_ir05(r#"{"format":"other/9"}"#).unwrap_err();
    assert!(err.contains("unsupported IR-05 format"));
}
