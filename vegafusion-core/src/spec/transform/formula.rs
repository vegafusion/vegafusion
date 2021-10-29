use crate::spec::transform::TransformSpecTrait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use crate::task_graph::task::InputVariable;
use crate::error::Result;
use crate::expression::parser::parse;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FormulaTransformSpec {
    pub expr: String,

    #[serde(rename = "as")]
    pub as_: String,

    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

impl TransformSpecTrait for FormulaTransformSpec {
    fn input_vars(&self) -> Result<Vec<InputVariable>> {
        let expr = parse(&self.expr)?;
        Ok(expr.input_vars())
    }
}