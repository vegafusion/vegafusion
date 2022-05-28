/*
 * VegaFusion
 * Copyright (C) 2022 VegaFusion Technologies LLC
 *
 * This program is distributed under multiple licenses.
 * Please consult the license documentation provided alongside
 * this program the details of the active license.
 */
use crate::spec::transform::{TransformColumns, TransformSpecTrait};
use crate::spec::values::Field;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use crate::expression::column_usage::{ColumnUsage, VlSelectionFields};
use crate::task_graph::graph::ScopedVariable;
use crate::task_graph::scope::TaskScope;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AggregateTransformSpec {
    pub groupby: Vec<Field>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub fields: Option<Vec<Option<Field>>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub ops: Option<Vec<AggregateOpSpec>>,

    #[serde(rename = "as", skip_serializing_if = "Option::is_none")]
    pub as_: Option<Vec<Option<String>>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub cross: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub drop: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub key: Option<Field>,

    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum AggregateOpSpec {
    Count,
    Valid,
    Missing,
    Distinct,
    Sum,
    Product,
    Mean,
    Average,
    Variance,
    Variancep,
    Stdev,
    Stdevp,
    Stderr,
    Median,
    Q1,
    Q3,
    Ci0,
    Ci1,
    Min,
    Max,
    Argmin,
    Argmax,
    Values,
}

impl AggregateOpSpec {
    pub fn name(&self) -> String {
        serde_json::to_value(self)
            .unwrap()
            .as_str()
            .unwrap()
            .to_string()
    }
}

impl TransformSpecTrait for AggregateTransformSpec {
    fn supported(&self) -> bool {
        // Check for supported aggregation op
        use AggregateOpSpec::*;
        let ops = self.ops.clone().unwrap_or_else(|| vec![Count]);
        for op in &ops {
            if !matches!(
                op,
                Count
                    | Valid
                    | Missing
                    | Distinct
                    | Sum
                    | Mean
                    | Average
                    | Min
                    | Max
                    | Variance
                    | Variancep
                    | Stdev
                    | Stdevp
            ) {
                // Unsupported aggregation op
                return false;
            }
        }

        // Cross aggregation not supported
        if let Some(true) = &self.cross {
            return false;
        }

        // drop=false not support
        if let Some(false) = &self.drop {
            return false;
        }
        true
    }

    fn transform_columns(
        &self,
        datum_var: &Option<ScopedVariable>,
        _usage_scope: &[u32],
        _task_scope: &TaskScope,
        _vl_selection_fields: &VlSelectionFields,
    ) -> TransformColumns {
        if let Some(datum_var) = datum_var {
            // Compute produced columns
            // Only handle the case where "as" contains a list of strings with length matching ops
            let ops = self.ops.clone().unwrap_or_else(|| vec![AggregateOpSpec::Count]);
            let as_: Vec<_> = self.as_.unwrap_or_default().iter().cloned().collect::<Option<Vec<_>>>().unwrap_or_default();
            let produced = if ops.len() == as_.len() {
                ColumnUsage::from(as_.as_slice())
            } else {
                ColumnUsage::Unknown
            };

            // Compute usaged columns
            // self.fi
            //
            // if let Some(as_) = &self.as_ {
            //
            // }
            //
            // ops.iter().enumerate().map(|(index, op)| {
            //     let new_col = self.as_
            //         .and_then(|as_| as_.get(index).cloned().flatten())
            //         .unwrap_or_else(|| {
            //             let field = self.
            //         })
            // })
            //
            // let bin_start = self.as_.and_then(|as_| as_.get(0).cloned()).unwrap_or_else(|| "unit0".to_string());
            // let mut produced_cols = vec![bin_start];
            //
            // if self.interval.unwrap_or(true) {
            //     let bin_end = self.as_.and_then(|as_| as_.get(1).cloned()).unwrap_or_else(|| "unit1".to_string());
            //     produced_cols.push(bin_end)
            // }
            //
            // let produced = ColumnUsage::from(produced_cols.as_slice());
            //
            // // Compute used columns
            // let field = self.field.field();
            // let col_usage = ColumnUsage::empty().with_column(&field);
            // let usage = DatasetsColumnUsage::empty().with_column_usage(
            //     datum_var, col_usage
            // );
            //
            // TransformColumns::PassThrough { usage, produced }
            todo!()
        } else {
            TransformColumns::Unknown
        }
    }
}
