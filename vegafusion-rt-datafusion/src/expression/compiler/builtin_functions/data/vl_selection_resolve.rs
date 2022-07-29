/*
 * VegaFusion
 * Copyright (C) 2022 VegaFusion Technologies LLC
 *
 * This program is distributed under multiple licenses.
 * Please consult the license documentation provided alongside
 * this program the details of the active license.
 */
use datafusion::arrow::datatypes::DataType;
use datafusion::logical_plan::{lit, DFSchema, Expr};
use datafusion::scalar::ScalarValue;
use itertools::Itertools;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::str::FromStr;
use vegafusion_core::arrow::datatypes::Field;
use vegafusion_core::data::scalar::ScalarValueHelpers;
use vegafusion_core::data::table::VegaFusionTable;
use vegafusion_core::error::{Result, VegaFusionError};
use vegafusion_core::proto::gen::expression::literal::Value;

use crate::expression::compiler::builtin_functions::data::vl_selection_test::{
    SelectionRow, SelectionType,
};

use vegafusion_core::proto::gen::{
    expression::expression::Expr as ProtoExpr, expression::Expression, expression::Literal,
};

use super::vl_selection_test::Op;

pub fn parse_args(args: &[Expression]) -> Result<Op> {
    let n = args.len();
    if !(0..=1).contains(&n) {
        return Err(VegaFusionError::internal(&format!(
            "vlSelectionResolve requires 1 or 2 arguments. Received {}",
            n
        )));
    }

    // Validate second argument and extract operation
    let op = if n < 1 {
        Op::Union
    } else {
        let arg1 = &args[0];
        match arg1.expr() {
            ProtoExpr::Literal(Literal { value: Some(Value::String(value)), .. }) => {
                // All good
                Op::from_str(value.as_str()).unwrap()
            }
            _ => {
                return Err(VegaFusionError::internal(&format!(
                    "The second argument to vlSelectionResolve, if provided, must be either 'union' or 'intersect'. \
                    Received {}", arg1
                )))
            }
        }
    };
    Ok(op)
}

pub fn vl_selection_resolve_fn(
    table: &VegaFusionTable,
    args: &[Expression],
    _schema: &DFSchema,
) -> Result<Expr> {
    // Validate args and get operation
    let _op = parse_args(args)?;

    // Extract vector of rows for selection dataset
    let rows = if let ScalarValue::List(Some(elements), _) = table.to_scalar_value()? {
        elements
    } else {
        unreachable!()
    };

    // let mut prop_names: Vec<String> = Vec::new();
    // let mut prop_values: Vec<ScalarValue> = Vec::new();

    let mut props: HashMap<String, Vec<ScalarValue>> = HashMap::new();

    for row in rows {
        let row_spec = SelectionRow::try_from(row)?;
        for (field, value) in row_spec.fields.iter().zip(&row_spec.values) {
            let value = match field.typ {
                SelectionType::Enum => {
                    if let ScalarValue::List(Some(elements), _) = value {
                        elements.clone()
                    } else {
                        vec![value.clone()]
                    }
                }
                _ => {
                    match &value {
                        ScalarValue::List(Some(elements), _) if elements.len() == 2 => {
                            // Don't assume elements are in ascending order
                            let first = elements[0].to_f64()?;
                            let second = elements[1].to_f64()?;

                            let (low, high) = if first <= second {
                                (first, second)
                            } else {
                                (second, first)
                            };
                            vec![ScalarValue::from(low), ScalarValue::from(high)]
                            // ScalarValue::List(Some(Box::new(vec![
                            //     ScalarValue::from(low), ScalarValue::from(high)
                            // ])), Box::new(DataType::Float64))
                        }
                        v => {
                            return Err(VegaFusionError::internal(&format!(
                                "values must be a two-element array. Found {}",
                                v
                            )))
                        }
                    }
                }
            };

            let values = props.entry(field.field.clone()).or_insert_with(Vec::new);
            values.extend(value.clone());
        }
    }

    let props: Vec<_> = props
        .into_iter()
        .map(|(name, values)| {
            // Turn values into a scalar list
            let dtype = values
                .get(0)
                .map(|s| s.get_datatype())
                .unwrap_or(DataType::Float64);
            let values = ScalarValue::List(Some(values), Box::new(Field::new("item", dtype, true)));
            (name, values)
        })
        .sorted_by_key(|(n, _)| n.clone())
        .collect();

    let props: Vec<_> = props
        .iter()
        .map(|(name, value)| (name.as_str(), value.clone()))
        .collect();

    let object_result = ScalarValue::from(props);
    Ok(lit(object_result))
}
