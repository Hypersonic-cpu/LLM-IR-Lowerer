# LLM IR Lowerer

A standalone Rust compiler for the layered LLM-serving IR defined by
`llm_serving_ir_design_v0_2.md`.

## Design authority

The GTSim v0.2 design document is normative. This repository is intentionally
independent of `GTSim/frontend`; that frontend is not reused as an implementation
or as a source of IR semantics.

Canonical interchange files and extensions follow the design document:
`.progir`, `.algoir`, `.flowir`, `.nvir`, and `.gtsir`. Payloads are
versioned and validated at the IR boundary.

IR-01 Program and IR-02 Algorithm Template are complementary inputs, not a
linear IR-01-to-IR-02 conversion. The IR-02-to-IR-03 lowering entry point is
deliberately unimplemented in the initial scope and must fail explicitly.

## Development

The implementation will be added in small, independently tested commits.
