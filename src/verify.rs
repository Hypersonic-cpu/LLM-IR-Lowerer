use crate::ir::*;
use std::collections::{BTreeMap, BTreeSet};

fn require(condition: bool, message: impl Into<String>) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

fn unique<I, S>(values: I, label: &str) -> Result<BTreeSet<String>, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut result = BTreeSet::new();
    for value in values {
        let value = value.into();
        require(!value.is_empty(), format!("empty {label}"))?;
        require(result.insert(value.clone()), format!("duplicate {label}: {value}"))?;
    }
    Ok(result)
}

fn check_acyclic(nodes: &BTreeSet<String>, edges: &[(String, String)]) -> Result<(), String> {
    let mut indegree: BTreeMap<String, usize> =
        nodes.iter().cloned().map(|node| (node, 0)).collect();
    let mut outgoing: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (from, to) in edges {
        require(nodes.contains(from), format!("dependency source not found: {from}"))?;
        require(nodes.contains(to), format!("dependency target not found: {to}"))?;
        require(from != to, format!("self dependency: {from}"))?;
        *indegree.get_mut(to).unwrap() += 1;
        outgoing.entry(from.clone()).or_default().push(to.clone());
    }
    let mut ready: Vec<String> = indegree
        .iter()
        .filter_map(|(node, degree)| (*degree == 0).then_some(node.clone()))
        .collect();
    let mut visited = 0usize;
    while let Some(node) = ready.pop() {
        visited += 1;
        if let Some(next) = outgoing.get(&node) {
            for successor in next {
                let degree = indegree.get_mut(successor).unwrap();
                *degree -= 1;
                if *degree == 0 {
                    ready.push(successor.clone());
                }
            }
        }
    }
    require(visited == nodes.len(), "dependency graph contains a cycle")
}

fn check_tensors(tensors: &[TensorDecl]) -> Result<BTreeSet<String>, String> {
    let ids = unique(tensors.iter().map(|tensor| tensor.id.clone()), "tensor id")?;
    unique(tensors.iter().map(|tensor| tensor.name.clone()), "tensor name")?;
    for tensor in tensors {
        require(!tensor.dtype.is_empty(), format!("tensor {} has no dtype", tensor.id))?;
        require(!tensor.storage.is_empty(), format!("tensor {} has no storage", tensor.id))?;
    }
    Ok(ids)
}

fn check_region(region: &TensorRegion, tensors: &BTreeSet<String>) -> Result<(), String> {
    require(
        tensors.contains(&region.tensor),
        format!("region references unknown tensor {}", region.tensor),
    )?;
    require(
        region.offsets.is_empty() || region.offsets.len() == region.shape.len(),
        format!("region {} has mismatched offset/shape rank", region.tensor),
    )
}

fn check_loop(loop_range: &LoopRange) -> Result<(), String> {
    require(loop_range.step > 0, format!("loop {} has zero step", loop_range.variable))?;
    require(
        loop_range.end >= loop_range.start,
        format!("loop {} has end before start", loop_range.variable),
    )
}

pub fn verify_program(program: &Program) -> Result<(), String> {
    require(program.format == PROGRAM_FORMAT, "invalid IR-01 format")?;
    require(!program.name.is_empty(), "program name is empty")?;
    let tensors = check_tensors(&program.tensors)?;
    let stream_names = unique(program.streams.iter().map(|stream| stream.name.clone()), "stream name")?;
    require(!stream_names.is_empty(), "program has no streams")?;
    for stream in &program.streams {
        let mut events = BTreeSet::new();
        let mut launches = BTreeSet::new();
        for command in &stream.commands {
            match command {
                ProgramCommand::Launch { invocation } => {
                    require(!invocation.callee.is_empty(), "launch has no callee")?;
                    require(launches.insert(invocation.id.clone()), format!("duplicate invocation {}", invocation.id))?;
                    require(!invocation.completion_event.is_empty(), "launch has no completion event")?;
                    for tensor in invocation.inputs.iter().chain(&invocation.outputs) {
                        require(tensors.contains(tensor), format!("launch {} references unknown tensor {tensor}", invocation.id))?;
                    }
                    for event in &invocation.after {
                        require(events.contains(event), format!("launch {} waits for unknown event {event}", invocation.id))?;
                    }
                    require(events.insert(invocation.completion_event.clone()), format!("duplicate event {}", invocation.completion_event))?;
                }
                ProgramCommand::RecordEvent { event } => {
                    require(!event.is_empty() && events.insert(event.clone()), format!("invalid/duplicate event {event}"))?;
                }
                ProgramCommand::WaitEvent { event } => {
                    require(events.contains(event), format!("wait before event record: {event}"))?;
                }
                ProgramCommand::Synchronize => {}
            }
        }
    }
    for tensor in &program.returns {
        require(tensors.contains(tensor), format!("return references unknown tensor {tensor}"))?;
    }
    Ok(())
}

pub fn verify_algorithm(algorithm: &AlgorithmTemplate) -> Result<(), String> {
    require(algorithm.format == ALGORITHM_FORMAT, "invalid IR-02 format")?;
    require(!algorithm.name.is_empty(), "algorithm name is empty")?;
    require(!algorithm.family.is_empty(), "algorithm has no implementation family")?;
    let domains = unique(algorithm.domains.iter().map(|domain| domain.name.clone()), "index domain")?;
    for domain in &algorithm.domains {
        if let Dimension::Static(extent) = &domain.extent {
            require(*extent > 0, format!("domain {} has zero extent", domain.name))?;
        }
    }
    let operands = unique(algorithm.operands.iter().map(|operand| operand.name.clone()), "operand")?;
    for operand in &algorithm.operands {
        require(!operand.dtype.is_empty(), format!("operand {} has no dtype", operand.name))?;
        for axis in &operand.axes {
            require(domains.contains(axis), format!("operand {} references unknown axis {axis}", operand.name))?;
        }
    }
    let mut values = operands;
    let mut outputs = BTreeSet::new();
    for operation in &algorithm.operations {
        require(!operation.id.is_empty() && !operation.kind.is_empty(), "invalid math operation")?;
        for input in &operation.inputs {
            require(values.contains(input), format!("operation {} reads undefined value {input}", operation.id))?;
        }
        require(outputs.insert(operation.output.clone()), format!("duplicate math output {}", operation.output))?;
        values.insert(operation.output.clone());
        for axis in operation.reduction_axes.iter().chain(&operation.output_axes) {
            require(domains.contains(axis), format!("operation {} references unknown axis {axis}", operation.id))?;
        }
    }
    let stages = unique(algorithm.stages.iter().map(|stage| stage.name.clone()), "algorithm stage")?;
    let mut edges = Vec::new();
    for stage in &algorithm.stages {
        for dependency in &stage.after {
            require(stages.contains(dependency), format!("stage {} depends on unknown stage {dependency}", stage.name))?;
            edges.push((dependency.clone(), stage.name.clone()));
        }
        for value in stage.reads.iter().chain(&stage.produces) {
            require(values.contains(value), format!("stage {} references unknown value {value}", stage.name))?;
        }
    }
    check_acyclic(&stages, &edges)
}

pub fn verify_flow(flow: &FlowKernel) -> Result<(), String> {
    require(flow.format == FLOW_FORMAT, "invalid IR-03 format")?;
    require(!flow.name.is_empty() && !flow.invocation_id.is_empty(), "flow identity is incomplete")?;
    require(flow.mapping.flow == flow.name, "mapping references a different flow")?;
    require(!flow.mapping.hardware.is_empty(), "mapping has no hardware profile")?;
    let tensors = check_tensors(&flow.tensors)?;
    let executors = unique(flow.executors.iter().map(|executor| executor.id.clone()), "executor")?;
    let buffers = unique(flow.buffers.iter().map(|buffer| buffer.id.clone()), "flow buffer")?;
    for executor in &flow.executors {
        require(executor.count > 0, format!("executor {} has zero count", executor.id))?;
    }
    for buffer in &flow.buffers {
        require(buffer.slots > 0, format!("buffer {} has zero slots", buffer.id))?;
    }
    let operations = unique(flow.operations.iter().map(|operation| operation.id.clone()), "flow operation")?;
    let mut edges = Vec::new();
    for operation in &flow.operations {
        require(executors.contains(&operation.executor), format!("operation {} uses unknown executor {}", operation.id, operation.executor))?;
        for dependency in &operation.dependencies {
            require(operations.contains(&dependency.operation), format!("operation {} depends on unknown operation {}", operation.id, dependency.operation))?;
            edges.push((dependency.operation.clone(), operation.id.clone()));
        }
        for region in operation.reads.iter().chain(&operation.writes) {
            check_region(region, &tensors)?;
        }
        for slot in &operation.slots {
            require(buffers.contains(&slot.buffer), format!("operation {} uses unknown buffer {}", operation.id, slot.buffer))?;
        }
        for loop_range in &operation.loops {
            check_loop(loop_range)?;
        }
    }
    check_acyclic(&operations, &edges)?;
    let cta = &flow.mapping.cta;
    require(cta.count > 0 && cta.warps > 0 && cta.threads > 0, "invalid CTA placement")?;
    require(cta.threads >= cta.warps * 32, "CTA thread count is smaller than its warp count")?;
    Ok(())
}

pub fn verify_hardware(profile: &HardwareProfile) -> Result<(), String> {
    require(profile.format == HARDWARE_PROFILE_FORMAT, "invalid hardware profile format")?;
    require(!profile.id.is_empty() && !profile.target.is_empty(), "hardware profile identity is incomplete")?;
    require(profile.max_shared_memory_bytes_per_cta > 0, "hardware profile has no shared memory")?;
    require(profile.max_registers_per_thread > 0 && profile.max_warps_per_cta > 0, "invalid hardware resource limits")?;
    unique(profile.memory_spaces.iter().cloned(), "memory space")?;
    let engines = unique(profile.engines.iter().map(|engine| engine.name.clone()), "engine")?;
    require(!engines.is_empty(), "hardware profile has no engines")?;
    for engine in &profile.engines {
        require(!engine.capabilities.is_empty(), format!("engine {} has no capabilities", engine.name))?;
        unique(engine.capabilities.iter().cloned(), "engine capability")?;
    }
    Ok(())
}

pub fn verify_arch(arch: &ArchKernel) -> Result<(), String> {
    require(arch.format == ARCH_FORMAT, "invalid IR-04 format")?;
    require(!arch.name.is_empty() && !arch.target.is_empty(), "ArchNative identity is incomplete")?;
    let tensors = check_tensors(&arch.tensors)?;
    let ctas = unique(arch.ctas.iter().map(|cta| cta.id.to_string()), "CTA id")?;
    require(!ctas.is_empty(), "ArchNative kernel has no CTA")?;
    for cta in &arch.ctas {
        require(cta.threads > 0 && cta.warps > 0, format!("CTA {} has invalid dimensions", cta.id))?;
    }
    let operations = unique(arch.operations.iter().map(|operation| operation.id.clone()), "native operation")?;
    let mut edges = Vec::new();
    for operation in &arch.operations {
        for dependency in &operation.dependencies {
            require(operations.contains(&dependency.operation), format!("native op {} depends on unknown op {}", operation.id, dependency.operation))?;
            edges.push((dependency.operation.clone(), operation.id.clone()));
        }
        for region in operation.reads.iter().chain(&operation.writes) {
            check_region(region, &tensors)?;
        }
        for loop_range in &operation.loops {
            check_loop(loop_range)?;
        }
    }
    check_acyclic(&operations, &edges)
}

pub fn verify_kernel_dag(kernel: &KernelDag) -> Result<(), String> {
    require(kernel.format == KERNEL_DAG_FORMAT, "invalid kernel DAG format")?;
    require(!kernel.name.is_empty() && !kernel.invocation_id.is_empty(), "kernel DAG identity is incomplete")?;
    let tensors = check_tensors(&kernel.tensors)?;
    let ctas = unique(kernel.ctas.iter().map(|cta| cta.id.to_string()), "CTA id")?;
    require(!ctas.is_empty(), "kernel DAG has no CTA")?;
    let nodes = unique(kernel.nodes.iter().map(|node| node.id.clone()), "DAG node")?;
    let mut edges = Vec::new();
    for node in &kernel.nodes {
        require(ctas.contains(&node.cta_id.to_string()), format!("node {} references unknown CTA {}", node.id, node.cta_id))?;
        for dependency in node.after_issue.iter().chain(&node.after_complete) {
            require(nodes.contains(dependency), format!("node {} depends on unknown node {dependency}", node.id))?;
            edges.push((dependency.clone(), node.id.clone()));
        }
        for region in node.reads.iter().chain(&node.writes) {
            check_region(region, &tensors)?;
        }
    }
    check_acyclic(&nodes, &edges)
}

pub fn verify_execution_program(program: &ExecutionProgram) -> Result<(), String> {
    require(program.format == EXECUTION_PROGRAM_FORMAT, "invalid execution program format")?;
    require(!program.name.is_empty(), "execution program name is empty")?;
    require(program.streams.len() == 1, "v0 execution supports exactly one ordered stream")?;
    let tensors = check_tensors(&program.tensors)?;
    let launches = unique(program.launches.iter().map(|launch| launch.id.clone()), "execution launch")?;
    let invocations = unique(program.launches.iter().map(|launch| launch.invocation_id.clone()), "invocation id")?;
    for launch in &program.launches {
        require(!launch.kernel_dag.is_empty(), format!("launch {} has no kernel DAG", launch.id))?;
        let mut local = BTreeSet::new();
        for binding in &launch.bindings {
            require(local.insert(binding.kernel_tensor_id.clone()), format!("duplicate local tensor binding {}", binding.kernel_tensor_id))?;
            require(tensors.contains(&binding.program_tensor_id), format!("binding references unknown program tensor {}", binding.program_tensor_id))?;
        }
    }
    let mut submitted = BTreeSet::new();
    let mut events = BTreeSet::new();
    for command in &program.commands {
        match command {
            ExecutionCommand::Launch { invocation_id, after, completion_event } => {
                require(invocations.contains(invocation_id), format!("unknown invocation {invocation_id}"))?;
                require(submitted.insert(invocation_id.clone()), format!("duplicate submitted invocation {invocation_id}"))?;
                for event in after {
                    require(events.contains(event), format!("launch waits for unknown event {event}"))?;
                }
                require(!completion_event.is_empty() && events.insert(completion_event.clone()), format!("invalid/duplicate completion event {completion_event}"))?;
            }
            ExecutionCommand::RecordEvent { event } => {
                require(!event.is_empty() && events.insert(event.clone()), format!("invalid/duplicate event {event}"))?;
            }
            ExecutionCommand::WaitEvent { event } => {
                require(events.contains(event), format!("wait before event record: {event}"))?;
            }
            ExecutionCommand::Synchronize => {}
        }
    }
    require(submitted == invocations, "not every execution launch is submitted")?;
    require(launches.len() == program.launches.len(), "duplicate execution launch")
}
