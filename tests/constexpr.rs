use llm_ir_lowerer::constexpr::*;
use llm_ir_lowerer::ir::*;
use serde_json::json;
use std::collections::BTreeMap;

fn launch(id: &str, batch: TemplateValue) -> TemplateCommand {
    TemplateCommand::Launch(TemplateInvocation {
        id: id.into(),
        callee: "gemm".into(),
        inputs: vec!["A".into()],
        outputs: vec!["C".into()],
        attrs: BTreeMap::from([("batch".into(), batch)]),
        after: vec![],
        completion_event: format!("e_{id}"),
    })
}

#[test]
fn static_if_selects_branch_and_substitutes_template_variables() {
    let env = BTreeMap::from([("B".into(), ConstValue::Int(4))]);
    let template = ProgramTemplate {
        name: "decode".into(),
        runtime_params: vec![RuntimeParam {
            name: "kv_lengths".into(),
            dtype: "vector<index>".into(),
        }],
        tensors: vec![TensorTemplate {
            id: "A".into(),
            name: "A".into(),
            shape: vec![TemplateValue::Variable("B".into()), TemplateValue::Literal(json!(128))],
            dtype: "bf16".into(),
            storage: "HBM".into(),
            external: true,
        }],
        stream: "compute".into(),
        commands: vec![TemplateCommand::StaticIf {
            condition: ConstExpr::Gt(
                Box::new(ConstExpr::Var("B".into())),
                Box::new(ConstExpr::Int(1)),
            ),
            then_body: vec![launch("large_batch", TemplateValue::Variable("B".into()))],
            else_body: vec![launch("small_batch", TemplateValue::Literal(json!(1)))],
        }],
        returns: vec![],
    };

    let program = template.instantiate(&env).unwrap();
    assert_eq!(program.format, PROGRAM_FORMAT);
    assert_eq!(program.tensors[0].shape, vec![Dimension::Static(4), Dimension::Static(128)]);
    let ProgramCommand::Launch { invocation } = &program.streams[0].commands[0] else {
        panic!("expected launch")
    };
    assert_eq!(invocation.id, "large_batch");
    assert_eq!(invocation.attrs["batch"], json!(4));
}

#[test]
fn static_if_requires_a_compile_time_boolean() {
    let env = BTreeMap::from([("B".into(), ConstValue::Int(2))]);
    let template = ProgramTemplate {
        name: "bad".into(),
        runtime_params: vec![],
        tensors: vec![],
        stream: "compute".into(),
        commands: vec![TemplateCommand::StaticIf {
            condition: ConstExpr::Var("B".into()),
            then_body: vec![],
            else_body: vec![],
        }],
        returns: vec![],
    };
    assert!(template.instantiate(&env).unwrap_err().contains("expected boolean"));
}

#[test]
fn compile_time_arithmetic_checks_overflow_and_division_by_zero() {
    let env = ConstEnv::new();
    let overflow = ConstExpr::Add(
        Box::new(ConstExpr::Int(i64::MAX)),
        Box::new(ConstExpr::Int(1)),
    );
    assert!(overflow.eval(&env).unwrap_err().contains("overflow"));
    let divide_by_zero = ConstExpr::Div(
        Box::new(ConstExpr::Int(1)),
        Box::new(ConstExpr::Int(0)),
    );
    assert!(divide_by_zero.eval(&env).unwrap_err().contains("division by zero"));
}

#[test]
fn canonical_program_dimensions_can_be_specialized() {
    let program = Program {
        format: PROGRAM_FORMAT.into(),
        name: "p".into(),
        params: vec![],
        tensors: vec![TensorDecl {
            id: "x".into(),
            name: "x".into(),
            shape: vec![Dimension::Runtime("B".into()), Dimension::Runtime("T".into())],
            dtype: "bf16".into(),
            storage: "HBM".into(),
            external: false,
        }],
        streams: vec![],
        returns: vec![],
    };
    let env = BTreeMap::from([("B".into(), ConstValue::Int(8))]);
    let specialized = specialize_program(&program, &env).unwrap();
    assert_eq!(specialized.tensors[0].shape, vec![Dimension::Static(8), Dimension::Runtime("T".into())]);
}
