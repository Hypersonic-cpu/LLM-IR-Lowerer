//! Compile-time template evaluation performed before canonical IR-01 emission.
//!
//! Template-only nodes are an in-memory construction API, not a new serialized
//! IR dialect. Instantiation emits the normal versioned .progir payload.

use crate::ir::{
    Dimension, KernelInvocation, Program, ProgramCommand, RuntimeParam, Stream, TensorDecl,
    PROGRAM_FORMAT,
};
use serde_json::Value;
use std::collections::BTreeMap;

pub type ConstEnv = BTreeMap<String, ConstValue>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstValue {
    Bool(bool),
    Int(i64),
    String(String),
}

impl ConstValue {
    fn to_json(&self) -> Value {
        match self {
            Self::Bool(v) => Value::Bool(*v),
            Self::Int(v) => Value::Number((*v).into()),
            Self::String(v) => Value::String(v.clone()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstExpr {
    Bool(bool),
    Int(i64),
    String(String),
    Var(String),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    Eq(Box<Self>, Box<Self>),
    Ne(Box<Self>, Box<Self>),
    Lt(Box<Self>, Box<Self>),
    Le(Box<Self>, Box<Self>),
    Gt(Box<Self>, Box<Self>),
    Ge(Box<Self>, Box<Self>),
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
    Not(Box<Self>),
}

impl ConstExpr {
    pub fn eval(&self, env: &ConstEnv) -> Result<ConstValue, String> {
        use ConstExpr::*;
        match self {
            Bool(v) => Ok(ConstValue::Bool(*v)),
            Int(v) => Ok(ConstValue::Int(*v)),
            String(v) => Ok(ConstValue::String(v.clone())),
            Var(name) => env
                .get(name)
                .cloned()
                .ok_or_else(|| format!("unknown compile-time variable {name}")),
            Add(a, b) => int_binary(a, b, env, "addition", i64::checked_add),
            Sub(a, b) => int_binary(a, b, env, "subtraction", i64::checked_sub),
            Mul(a, b) => int_binary(a, b, env, "multiplication", i64::checked_mul),
            Div(a, b) => {
                let lhs = eval_int(a, env)?;
                let rhs = eval_int(b, env)?;
                if rhs == 0 {
                    return Err("compile-time division by zero".into());
                }
                lhs.checked_div(rhs)
                    .map(ConstValue::Int)
                    .ok_or_else(|| "compile-time integer division overflow".into())
            }
            Eq(a, b) => Ok(ConstValue::Bool(a.eval(env)? == b.eval(env)?)),
            Ne(a, b) => Ok(ConstValue::Bool(a.eval(env)? != b.eval(env)?)),
            Lt(a, b) => compare_int(a, b, env, |x, y| x < y),
            Le(a, b) => compare_int(a, b, env, |x, y| x <= y),
            Gt(a, b) => compare_int(a, b, env, |x, y| x > y),
            Ge(a, b) => compare_int(a, b, env, |x, y| x >= y),
            And(a, b) => {
                let lhs = eval_bool(a, env)?;
                if !lhs {
                    Ok(ConstValue::Bool(false))
                } else {
                    Ok(ConstValue::Bool(eval_bool(b, env)?))
                }
            }
            Or(a, b) => {
                let lhs = eval_bool(a, env)?;
                if lhs {
                    Ok(ConstValue::Bool(true))
                } else {
                    Ok(ConstValue::Bool(eval_bool(b, env)?))
                }
            }
            Not(value) => Ok(ConstValue::Bool(!eval_bool(value, env)?)),
        }
    }
}

fn eval_int(expr: &ConstExpr, env: &ConstEnv) -> Result<i64, String> {
    match expr.eval(env)? {
        ConstValue::Int(value) => Ok(value),
        other => Err(format!("expected integer constant, got {other:?}")),
    }
}

fn eval_bool(expr: &ConstExpr, env: &ConstEnv) -> Result<bool, String> {
    match expr.eval(env)? {
        ConstValue::Bool(value) => Ok(value),
        other => Err(format!("expected boolean constant, got {other:?}")),
    }
}

fn int_binary(
    lhs: &ConstExpr,
    rhs: &ConstExpr,
    env: &ConstEnv,
    label: &str,
    operation: fn(i64, i64) -> Option<i64>,
) -> Result<ConstValue, String> {
    operation(eval_int(lhs, env)?, eval_int(rhs, env)?)
        .map(ConstValue::Int)
        .ok_or_else(|| format!("compile-time {label} overflow"))
}

fn compare_int(
    lhs: &ConstExpr,
    rhs: &ConstExpr,
    env: &ConstEnv,
    compare: fn(i64, i64) -> bool,
) -> Result<ConstValue, String> {
    Ok(ConstValue::Bool(compare(
        eval_int(lhs, env)?,
        eval_int(rhs, env)?,
    )))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TemplateValue {
    Literal(Value),
    Variable(String),
    Expression(ConstExpr),
}

impl TemplateValue {
    fn resolve(&self, env: &ConstEnv) -> Result<Value, String> {
        match self {
            Self::Literal(value) => Ok(value.clone()),
            Self::Variable(name) => env
                .get(name)
                .map(ConstValue::to_json)
                .ok_or_else(|| format!("unknown compile-time variable {name}")),
            Self::Expression(expr) => Ok(expr.eval(env)?.to_json()),
        }
    }

    fn resolve_dimension(&self, env: &ConstEnv) -> Result<Dimension, String> {
        match self.resolve(env)? {
            Value::Number(value) => value
                .as_u64()
                .map(Dimension::Static)
                .ok_or_else(|| "tensor dimensions must be non-negative integers".into()),
            Value::String(value) => Ok(Dimension::Runtime(value)),
            other => Err(format!("invalid tensor dimension template value {other}")),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorTemplate {
    pub id: String,
    pub name: String,
    pub shape: Vec<TemplateValue>,
    pub dtype: String,
    pub storage: String,
    pub external: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemplateInvocation {
    pub id: String,
    pub callee: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: BTreeMap<String, TemplateValue>,
    pub after: Vec<String>,
    pub completion_event: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TemplateCommand {
    Launch(TemplateInvocation),
    RecordEvent(String),
    WaitEvent(String),
    Synchronize,
    StaticIf {
        condition: ConstExpr,
        then_body: Vec<TemplateCommand>,
        else_body: Vec<TemplateCommand>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramTemplate {
    pub name: String,
    pub runtime_params: Vec<RuntimeParam>,
    pub tensors: Vec<TensorTemplate>,
    pub stream: String,
    pub commands: Vec<TemplateCommand>,
    pub returns: Vec<String>,
}

impl ProgramTemplate {
    /// Resolve all static branches and template values, producing canonical IR-01.
    pub fn instantiate(&self, env: &ConstEnv) -> Result<Program, String> {
        let mut commands = Vec::new();
        expand_commands(&self.commands, env, &mut commands)?;

        let tensors = self
            .tensors
            .iter()
            .map(|tensor| {
                Ok(TensorDecl {
                    id: tensor.id.clone(),
                    name: tensor.name.clone(),
                    shape: tensor
                        .shape
                        .iter()
                        .map(|dimension| dimension.resolve_dimension(env))
                        .collect::<Result<Vec<_>, _>>()?,
                    dtype: tensor.dtype.clone(),
                    storage: tensor.storage.clone(),
                    external: tensor.external,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        Ok(Program {
            format: PROGRAM_FORMAT.into(),
            name: self.name.clone(),
            params: self.runtime_params.clone(),
            tensors,
            streams: vec![Stream {
                name: self.stream.clone(),
                ordered: true,
                commands,
            }],
            returns: self.returns.clone(),
        })
    }
}

fn expand_commands(
    input: &[TemplateCommand],
    env: &ConstEnv,
    output: &mut Vec<ProgramCommand>,
) -> Result<(), String> {
    for command in input {
        match command {
            TemplateCommand::Launch(invocation) => {
                let attrs = invocation
                    .attrs
                    .iter()
                    .map(|(key, value)| Ok((key.clone(), value.resolve(env)?)))
                    .collect::<Result<_, String>>()?;
                output.push(ProgramCommand::Launch {
                    invocation: KernelInvocation {
                        id: invocation.id.clone(),
                        callee: invocation.callee.clone(),
                        inputs: invocation.inputs.clone(),
                        outputs: invocation.outputs.clone(),
                        attrs,
                        after: invocation.after.clone(),
                        completion_event: invocation.completion_event.clone(),
                    },
                });
            }
            TemplateCommand::RecordEvent(event) => {
                output.push(ProgramCommand::RecordEvent {
                    event: event.clone(),
                });
            }
            TemplateCommand::WaitEvent(event) => {
                output.push(ProgramCommand::WaitEvent {
                    event: event.clone(),
                });
            }
            TemplateCommand::Synchronize => output.push(ProgramCommand::Synchronize),
            TemplateCommand::StaticIf {
                condition,
                then_body,
                else_body,
            } => {
                if eval_bool(condition, env)? {
                    expand_commands(then_body, env, output)?;
                } else {
                    expand_commands(else_body, env, output)?;
                }
            }
        }
    }
    Ok(())
}

/// Specialize symbolic dimensions in an already canonical IR-01 document.
pub fn specialize_program(program: &Program, env: &ConstEnv) -> Result<Program, String> {
    let mut specialized = program.clone();
    for tensor in &mut specialized.tensors {
        for dimension in &mut tensor.shape {
            if let Dimension::Runtime(symbol) = dimension {
                if let Some(value) = env.get(symbol) {
                    *dimension = match value {
                        ConstValue::Int(value) if *value >= 0 => {
                            Dimension::Static(*value as u64)
                        }
                        ConstValue::String(value) => Dimension::Runtime(value.clone()),
                        other => {
                            return Err(format!(
                                "dimension symbol {symbol} requires a non-negative integer, got {other:?}"
                            ));
                        }
                    };
                }
            }
        }
    }
    Ok(specialized)
}
