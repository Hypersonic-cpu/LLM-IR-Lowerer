//! Standalone compiler for the layered LLM-serving IR.
//!
//! The v0.2 design document is normative. IR-01 and IR-02 are complementary
//! inputs; they are not a linear IR-01 -> IR-02 conversion.

pub mod codec;
pub mod constexpr;
pub mod ir;
pub mod verify;

pub use ir::*;
