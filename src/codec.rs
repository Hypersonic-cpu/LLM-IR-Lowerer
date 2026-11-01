use crate::ir::{
    ArchKernel, AlgorithmTemplate, ExecutionProgram, FlowKernel, HardwareProfile, Ir05Document,
    KernelDag, Program, ARCH_FORMAT, ALGORITHM_FORMAT, EXECUTION_PROGRAM_FORMAT, FLOW_FORMAT,
    HARDWARE_PROFILE_FORMAT, KERNEL_DAG_FORMAT, PROGRAM_FORMAT,
};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;
use std::fs;
use std::path::Path;

pub fn encode_pretty<T: Serialize>(value: &T) -> Result<String, String> {
    serde_json::to_string_pretty(value).map_err(|e| format!("serialize IR: {e}"))
}

pub fn decode<T: DeserializeOwned>(text: &str) -> Result<T, String> {
    serde_json::from_str(text).map_err(|e| format!("parse IR JSON: {e}"))
}

pub fn read<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T, String> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    decode(&text)
}

pub fn write<T: Serialize>(path: impl AsRef<Path>, value: &T) -> Result<(), String> {
    let path = path.as_ref();
    let text = encode_pretty(value)?;
    fs::write(path, format!("{text}\n"))
        .map_err(|e| format!("write {}: {e}", path.display()))
}

pub fn decode_ir05(text: &str) -> Result<Ir05Document, String> {
    let value: Value = decode(text)?;
    let format = value
        .get("format")
        .and_then(Value::as_str)
        .ok_or_else(|| "IR-05 document is missing string field format".to_string())?;
    match format {
        KERNEL_DAG_FORMAT => decode::<KernelDag>(text).map(Ir05Document::Kernel),
        EXECUTION_PROGRAM_FORMAT => {
            decode::<ExecutionProgram>(text).map(Ir05Document::Program)
        }
        other => Err(format!("unsupported IR-05 format {other}")),
    }
}

pub fn verify_format<T: Serialize>(value: &T, expected: &str) -> Result<(), String> {
    let value = serde_json::to_value(value).map_err(|e| e.to_string())?;
    let actual = value
        .get("format")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing string field format".to_string())?;
    if actual != expected {
        return Err(format!("format mismatch: expected {expected}, got {actual}"));
    }
    Ok(())
}

pub fn verify_json_file(path: impl AsRef<Path>) -> Result<String, String> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    match path.extension().and_then(|x| x.to_str()).unwrap_or("") {
        "progir" => {
            let ir: Program = decode(&text)?;
            verify_format(&ir, PROGRAM_FORMAT)?;
            Ok("IR-01 Program".into())
        }
        "algoir" => {
            let ir: AlgorithmTemplate = decode(&text)?;
            verify_format(&ir, ALGORITHM_FORMAT)?;
            Ok("IR-02 Algorithm Template".into())
        }
        "flowir" => {
            let ir: FlowKernel = decode(&text)?;
            verify_format(&ir, FLOW_FORMAT)?;
            Ok("IR-03 Scheduled Flow".into())
        }
        "nvir" => {
            let ir: ArchKernel = decode(&text)?;
            verify_format(&ir, ARCH_FORMAT)?;
            Ok("IR-04 ArchNative".into())
        }
        "gtsir" => match decode_ir05(&text)? {
            Ir05Document::Kernel(_) => Ok("IR-05 Kernel DAG".into()),
            Ir05Document::Program(_) => Ok("IR-05 Execution Program".into()),
        },
        "json" => {
            let hw: HardwareProfile = decode(&text)?;
            verify_format(&hw, HARDWARE_PROFILE_FORMAT)?;
            Ok("Hardware Profile".into())
        }
        ext => Err(format!("unsupported file extension .{ext}")),
    }
}
