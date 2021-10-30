
use datafusion::prelude::col;
use datafusion::logical_plan::{Expr, lit};
use std::str::FromStr;
use std::convert::TryFrom;
use vegafusion_core::data::scalar::ScalarValue;
use vegafusion_core::error::{Result, VegaFusionError, ResultWithContext};
use datafusion::logical_plan::DFSchema;
use crate::expression::compiler::utils::cast_to;
use std::collections::HashMap;
use vegafusion_core::proto::gen::{
    expression::Expression,
    expression::expression::Expr as ProtoExpr,
    expression::Literal,
    expression::literal::Value as ProtoValue,
    expression::Identifier,
};
use vegafusion_core::expression::ast::expression;
use vegafusion_core::proto::gen::expression::literal::Value;
use vegafusion_core::data::table::VegaFusionTable;

/// Op
#[derive(Debug, Clone)]
enum Op {
    Union,
    Intersect,
}

impl Op {
    pub fn valid(s: &str) -> bool {
        Self::from_str(s).is_ok()
    }
}

impl FromStr for Op {
    type Err = VegaFusionError;

    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "union" => Self::Union,
            "intersect" => Self::Intersect,
            _ => return Err(VegaFusionError::internal(&format!("Invalid vlSelection operation: {}", s))),
        })
    }
}

impl TryFrom<ScalarValue> for Op {
    type Error = VegaFusionError;

    fn try_from(value: ScalarValue) -> Result<Self> {
        match value {
            ScalarValue::Utf8(Some(op)) => Self::from_str(&op),
            _ => return Err(VegaFusionError::internal("Expected selection op to be a string")),
        }
    }
}

/// Selection Type
#[derive(Debug, Clone)]
enum SelectionType {
    Enum,
    RangeInc,
    RangeExc,
    RangeLe,
    RangeRe,
}

impl FromStr for SelectionType {
    type Err = VegaFusionError;

    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "E" => Self::Enum,
            "R" => Self::RangeInc,
            "R-E" => Self::RangeExc,
            "R-LE" => Self::RangeLe,
            "R-RE" => Self::RangeRe,
            _ => return Err(VegaFusionError::internal(&format!("Invalid selection type: {}", s))),
        })
    }
}

impl TryFrom<ScalarValue> for SelectionType {
    type Error = VegaFusionError;

    fn try_from(value: ScalarValue) -> Result<Self> {
        match value {
            ScalarValue::Utf8(Some(op)) => Self::from_str(&op),
            _ => return Err(VegaFusionError::internal("Expected selection type to be a string")),
        }
    }
}

/// Field specification
#[derive(Debug, Clone)]
struct FieldSpec {
    field: String,
    typ: SelectionType,
}

impl FieldSpec {
    pub fn to_expr(&self, values: &ScalarValue, schema: &DFSchema) -> Result<Expr> {
        let field_col = col(&self.field);
        let expr = match self.typ {
            SelectionType::Enum => {
                let list_values: Vec<_> = if let ScalarValue::List(Some(elements), _) = &values {
                    // values already a list
                    elements.iter().map(|el| lit(el.clone())).collect()
                } else {
                    // convert values to single element list
                    vec![lit(values.clone())]
                };
                Expr::InList {
                    expr: Box::new(field_col),
                    list: list_values,
                    negated: false,
                }
            }
            _ => {
                let field_dtype = field_col
                    .get_type(schema)
                    .with_context(|| format!("Failed to infer type of column {}", self.field))?;

                let (low, high) = match &values {
                    ScalarValue::List(Some(elements), _) if elements.len() == 2 => {
                        (lit(elements[0].clone()), lit(elements[1].clone()))
                    }
                    v => return Err(VegaFusionError::internal(
                        &format!("values must be a two-element array. Found {}", v)
                    )),
                };

                // Cast low/high scalar values to match the type of the field they will be compared to
                // Motivation: when field_dtype is Int64, and low/high are Float64, then without
                // casting, DataFusion will convert the whole field column to Float64 before running
                // the comparison.
                // We may need to revisit potential numerical precision issues at the boundaries
                let low = cast_to(low, &field_dtype, schema)?;
                let high = cast_to(high, &field_dtype, schema)?;

                match self.typ {
                    SelectionType::RangeInc => Expr::Between {
                        expr: Box::new(field_col),
                        negated: false,
                        low: Box::new(low),
                        high: Box::new(high),
                    },
                    SelectionType::RangeExc => low.lt(field_col.clone()).and(field_col.lt(high)),
                    SelectionType::RangeLe => low.lt(field_col.clone()).and(field_col.lt_eq(high)),
                    SelectionType::RangeRe => low.lt_eq(field_col.clone()).and(field_col.lt(high)),
                    SelectionType::Enum => {
                        unreachable!()
                    }
                }
            }
        };

        Ok(expr)
    }
}

impl TryFrom<ScalarValue> for FieldSpec {
    type Error = VegaFusionError;

    fn try_from(value: ScalarValue) -> Result<Self> {
        match value {
            ScalarValue::Struct(Some(values), fields) => {
                let field_names: HashMap<_, _> = fields
                    .iter()
                    .enumerate()
                    .map(|(ind, f)| (f.name().clone(), ind))
                    .collect();

                // Parse field
                let field_index = field_names
                    .get("field")
                    .with_context(|| "Missing required property 'field'".to_string())?;

                let field = match values.get(*field_index) {
                    Some(ScalarValue::Utf8(Some(field))) => field.clone(),
                    _ => return Err(VegaFusionError::internal(&format!("Expected field to be a string"))),
                };

                // Parse type
                let typ_index = field_names
                    .get("type")
                    .with_context(|| "Missing required property 'type'".to_string())?;
                let typ = SelectionType::try_from(values.get(*typ_index).unwrap().clone())?;

                Ok(Self { field, typ })
            }
            _ => return Err(VegaFusionError::internal(
                &format!("Expected selection field specification to be an object")
            )),
        }
    }
}

/// Selection row
pub struct SelectionRow {
    fields: Vec<FieldSpec>,
    values: Vec<ScalarValue>,
}

impl SelectionRow {
    pub fn to_expr(&self, schema: &DFSchema) -> Result<Expr> {
        let mut exprs: Vec<Expr> = Vec::new();
        for (field, value) in self.fields.iter().zip(self.values.iter()) {
            exprs.push(field.to_expr(value, schema)?);
        }

        // Take conjunction of expressions
        Ok(exprs.into_iter().reduce(|a, b| a.and(b)).unwrap())
    }
}

impl TryFrom<ScalarValue> for SelectionRow {
    type Error = VegaFusionError;

    fn try_from(value: ScalarValue) -> Result<Self> {
        match value {
            ScalarValue::Struct(Some(struct_values), struct_fields) => {
                let field_names: HashMap<_, _> = struct_fields
                    .iter()
                    .enumerate()
                    .map(|(ind, f)| (f.name().clone(), ind))
                    .collect();

                // Parse values
                let values_index = field_names
                    .get("values")
                    .with_context(|| "Missing required property 'values'".to_string())?;
                let values = match struct_values.get(*values_index) {
                    Some(ScalarValue::List(Some(elements), _)) => elements.as_ref().clone(),
                    _ => return Err(VegaFusionError::internal(&format!("Expected 'values' to be an array"))),
                };

                // Parse fields
                let fields_index = field_names
                    .get("fields")
                    .with_context(|| "Missing required property 'fields'".to_string())?;

                let mut fields: Vec<FieldSpec> = Vec::new();
                match struct_values.get(*fields_index) {
                    Some(ScalarValue::List(Some(elements), _)) => {
                        for el in elements.iter() {
                            fields.push(FieldSpec::try_from(el.clone())?)
                        }
                    }
                    _ => return Err(VegaFusionError::internal("Expected 'values' to be an array")),
                };

                // Validate lengths
                if values.len() != fields.len() {
                    return Err(VegaFusionError::internal(&format!(
                        "Length of selection fields ({}) must match that of selection values ({})",
                        fields.len(),
                        values.len()
                    )))
                }

                if values.is_empty() {
                    return Err(VegaFusionError::internal("Selection fields not be empty"))
                }

                Ok(Self { values, fields })
            }
            _ => return Err(VegaFusionError::internal("Expected selection row specification to be an object")),
        }
    }
}

fn parse_args(args: &[Expression]) -> Result<Op> {
    let n = args.len();
    if !(1..=2).contains(&n) {
        return Err(VegaFusionError::internal(
            &format!("vlSelectionTest requires 2 or 3 arguments. Received {}", n)
        ))
    }

    // Validate second argument
    // ProtoExpr::Identifier(Indentifier)
    match &args[0].expr() {
        ProtoExpr::Identifier(ident) if ident.name == "datum" => {
            // All good
        }
        arg => {
            return Err(VegaFusionError::internal(&format!(
                "The second argument to vlSelectionTest must be datum. Received {:?}",
                arg
            )))
        }
    }

    // Validate third argument and extract operation
    let op = if n < 2 {
        Op::Union
    } else {
        let arg1 = &args[1];
        match arg1.expr() {
            ProtoExpr::Literal(Literal { value: Some(Value::String(value)), .. }) => {
                // All good
                Op::from_str(value.as_str()).unwrap()
            }
            _ => {
                return Err(VegaFusionError::internal(&format!(
                    "The third argument to vlSelectionTest, if provided, must be either 'union' or 'intersect'. \
                    Received {}", arg1
                )))
            }
        }
    };
    Ok(op)
}

pub fn vl_selection_test_fn(
    table: &VegaFusionTable,
    args: &[Expression],
    schema: &DFSchema,
) -> Result<Expr> {
    // Validate args and get operation
    let op = parse_args(args)?;

    // Extract vector of rows for selection dataset
    let rows = if let ScalarValue::List(Some(elements), _) = table.to_scalar_value()? {
        elements.as_ref().clone()
    } else {
        unreachable!()
    };

    // Calculate selection expression for each row in selection dataset
    let mut exprs: Vec<Expr> = Vec::new();
    for row in rows {
        let row_spec = SelectionRow::try_from(row)?;
        exprs.push(row_spec.to_expr(schema)?)
    }

    // Combine expressions according to op
    let expr = if exprs.is_empty() {
        lit(false)
    } else {
        match op {
            Op::Union => exprs.into_iter().reduce(|a, b| a.or(b)).unwrap(),
            Op::Intersect => exprs.into_iter().reduce(|a, b| a.and(b)).unwrap(),
        }
    };

    Ok(expr)
}