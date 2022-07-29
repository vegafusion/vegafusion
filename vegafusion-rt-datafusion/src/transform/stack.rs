/*
 * VegaFusion
 * Copyright (C) 2022 VegaFusion Technologies LLC
 *
 * This program is distributed under multiple licenses.
 * Please consult the license documentation provided alongside
 * this program the details of the active license.
 */
use crate::expression::compiler::config::CompilationConfig;
use crate::expression::compiler::utils::to_numeric;
use crate::transform::TransformTrait;
use async_trait::async_trait;
use datafusion::dataframe::DataFrame;
use datafusion::physical_plan::aggregates;
use datafusion_expr::logical_plan::JoinType;
use datafusion_expr::{
    abs, col, lit, max, when, AggregateFunction, BuiltInWindowFunction, Expr, WindowFunction,
};
use std::ops::{Add, Div, Sub};
use std::sync::Arc;
use vegafusion_core::error::{Result, VegaFusionError};
use vegafusion_core::proto::gen::transforms::{SortOrder, Stack, StackOffset};
use vegafusion_core::task_graph::task_value::TaskValue;

#[async_trait]
impl TransformTrait for Stack {
    async fn eval(
        &self,
        dataframe: Arc<DataFrame>,
        _config: &CompilationConfig,
    ) -> Result<(Arc<DataFrame>, Vec<TaskValue>)> {
        // Assume offset is Zero until others are implemented
        let alias0 = self.alias_0.clone().expect("alias0 expected");
        let alias1 = self.alias_1.clone().expect("alias1 expected");

        // Save off input columns
        let input_fields: Vec<_> = dataframe
            .schema()
            .fields()
            .iter()
            .map(|f| f.field().name().clone())
            .collect();

        // Build order by vector
        // Order by row number last (and only if no explicit ordering provided)
        let mut order_by: Vec<_> = self
            .sort_fields
            .iter()
            .zip(&self.sort)
            .map(|(field, order)| Expr::Sort {
                expr: Box::new(col(field)),
                asc: *order == SortOrder::Ascending as i32,
                nulls_first: *order == SortOrder::Ascending as i32,
            })
            .collect();

        order_by.push(Expr::Sort {
            expr: Box::new(col("__row_number")),
            asc: true,
            nulls_first: true,
        });

        // Add row number column for sorting
        let row_number_expr = Expr::WindowFunction {
            fun: WindowFunction::BuiltInWindowFunction(BuiltInWindowFunction::RowNumber),
            args: Vec::new(),
            partition_by: Vec::new(),
            order_by: Vec::new(),
            window_frame: None,
        }
        .alias("__row_number");

        let dataframe = dataframe.select(vec![Expr::Wildcard, row_number_expr])?;

        // Process according to offset
        let offset = StackOffset::from_i32(self.offset).expect("Failed to convert stack offset");
        let dataframe = match offset {
            StackOffset::Zero => eval_zero_offset(
                self,
                dataframe,
                input_fields.as_slice(),
                &alias0,
                &alias1,
                order_by.as_slice(),
            )?,
            StackOffset::Normalize => eval_normalize_center_offset(
                self,
                dataframe,
                input_fields.as_slice(),
                &alias0,
                &alias1,
                order_by.as_slice(),
                &offset,
            )?,
            StackOffset::Center => eval_normalize_center_offset(
                self,
                dataframe,
                input_fields.as_slice(),
                &alias0,
                &alias1,
                order_by.as_slice(),
                &offset,
            )?,
        };

        Ok((dataframe, Default::default()))
    }
}

fn eval_normalize_center_offset(
    stack: &Stack,
    dataframe: Arc<DataFrame>,
    input_fields: &[String],
    alias0: &str,
    alias1: &str,
    order_by: &[Expr],
    offset: &StackOffset,
) -> Result<Arc<DataFrame>> {
    // Build first selection
    let mut selection_0: Vec<_> = input_fields
        .iter()
        .filter_map(|field| {
            if field == alias0 || field == alias1 {
                None
            } else {
                Some(col(field))
            }
        })
        .collect();
    selection_0.push(col("__row_number"));

    // Build groupby columns
    let partition_by: Vec<_> = stack.groupby.iter().map(|group| col(group)).collect();

    // Build a dataframe that holds the sum of each partition

    // Build window expression
    let fun = WindowFunction::AggregateFunction(aggregates::AggregateFunction::Sum);

    // Cast field to number, replace with 0 when null, take absolute value
    let numeric_field = to_numeric(col(&stack.field), dataframe.schema())?;
    let numeric_field = when(col(&stack.field).is_not_null(), numeric_field).otherwise(lit(0))?;
    let numeric_field = abs(numeric_field);

    let total_agg = Expr::AggregateFunction {
        fun: AggregateFunction::Sum,
        args: vec![numeric_field.clone()],
        distinct: false,
    }
    .alias("__total");

    let dataframe = if partition_by.is_empty() {
        let total_dataframe = dataframe.aggregate(partition_by.clone(), vec![total_agg])?;
        let total_dataframe =
            total_dataframe.select(vec![Expr::Wildcard, lit(true).alias("__unit_lhs")])?;
        let dataframe = dataframe.select(vec![Expr::Wildcard, lit(true).alias("__unit_rhs")])?;

        dataframe.join(
            total_dataframe,
            JoinType::Full,
            vec!["__unit_lhs"].as_slice(),
            vec!["__unit_rhs"].as_slice(),
            None,
        )?
    } else {
        let total_dataframe = dataframe.aggregate(partition_by.clone(), vec![total_agg])?;

        // Rename group cols prior to join
        let groupby_aliases: Vec<String> = partition_by
            .iter()
            .enumerate()
            .map(|(i, _a)| format!("grp_field_{:?}{}", stack.groupby, i))
            .collect();

        let mut select_exprs = vec![col("__total")];
        select_exprs.extend(
            partition_by
                .clone()
                .into_iter()
                .zip(&groupby_aliases)
                .map(|(n, alias)| n.alias(alias)),
        );
        let total_dataframe = total_dataframe.select(select_exprs)?;

        let left_cols: Vec<_> = stack.groupby.iter().map(|name| name.as_str()).collect();
        let right_cols: Vec<_> = groupby_aliases.iter().map(|name| name.as_str()).collect();
        dataframe.join(
            total_dataframe,
            JoinType::Inner,
            left_cols.as_slice(),
            right_cols.as_slice(),
            None,
        )?
    };

    // Build window function
    let window_expr = Expr::WindowFunction {
        fun,
        args: vec![numeric_field.clone()],
        partition_by,
        order_by: Vec::from(order_by),
        window_frame: None,
    }
    .alias(alias1);

    selection_0.push(window_expr);
    selection_0.push(col("__total"));

    // Perform selection to add new field value
    let dataframe = dataframe.select(selection_0.clone())?;

    // Restore original order
    let dataframe = dataframe.sort(vec![Expr::Sort {
        expr: Box::new(col("__row_number")),
        asc: true,
        nulls_first: false,
    }])?;

    // Build selection_1
    let mut selection_1: Vec<_> = input_fields
        .iter()
        .filter_map(|field| {
            if field == alias0 || field == alias1 {
                None
            } else {
                Some(col(field))
            }
        })
        .collect();

    // Now compute alias1 column by adding numeric field to alias0
    let dataframe = match offset {
        StackOffset::Center => {
            // Center bars between 0 and the max total length bar
            let max_total = max(col("__total")).alias("__max_total");
            let max_total_dataframe = dataframe
                .aggregate(vec![lit(true)], vec![max_total])?
                .select(vec![Expr::Wildcard, lit(true).alias("__max_unit_rhs")])?;

            let dataframe = dataframe
                .select(vec![Expr::Wildcard, lit(true).alias("__max_unit_lhs")])?
                .join(
                    max_total_dataframe,
                    JoinType::Inner,
                    vec!["__max_unit_lhs"].as_slice(),
                    vec!["__max_unit_rhs"].as_slice(),
                    None,
                )?;

            let first = col("__max_total").sub(col("__total")).div(lit(2));
            let alias1_col = col(alias1).add(first).alias(alias1);
            let alias0_col = alias1_col.clone().sub(numeric_field).alias(alias0);
            selection_1.push(alias0_col);
            selection_1.push(alias1_col);

            println!("{:#?}", dataframe.schema());

            dataframe
        }
        StackOffset::Normalize => {
            let alias0_col = col(alias1)
                .sub(numeric_field)
                .div(col("__total"))
                .alias(alias0);
            selection_1.push(alias0_col);

            let alias1_col = col(alias1).div(col("__total")).alias(alias1);
            selection_1.push(alias1_col);

            dataframe
        }
        _ => return Err(VegaFusionError::internal("Unexpected stack offset")),
    };

    let dataframe = dataframe.select(selection_1.clone())?;
    Ok(dataframe)
}

fn eval_zero_offset(
    stack: &Stack,
    dataframe: Arc<DataFrame>,
    input_fields: &[String],
    alias0: &str,
    alias1: &str,
    order_by: &[Expr],
) -> Result<Arc<DataFrame>> {
    // Build first selection
    let mut selection_0: Vec<_> = input_fields
        .iter()
        .filter_map(|field| {
            if field == alias0 || field == alias1 {
                None
            } else {
                Some(col(field))
            }
        })
        .collect();
    selection_0.push(col("__row_number"));

    // Build groupby columns
    let partition_by: Vec<_> = stack.groupby.iter().map(|group| col(group)).collect();

    // Build window expression
    let fun = WindowFunction::AggregateFunction(aggregates::AggregateFunction::Sum);

    // Cast field to number and replace with 0 when null
    let numeric_field = to_numeric(col(&stack.field), dataframe.schema())?;
    let numeric_field = when(col(&stack.field).is_not_null(), numeric_field).otherwise(lit(0))?;

    let window_expr = Expr::WindowFunction {
        fun,
        args: vec![numeric_field.clone()],
        partition_by,
        order_by: Vec::from(order_by),
        window_frame: None,
    }
    .alias(alias1);

    selection_0.push(window_expr);

    // For offset zero, we need to evaluate positive and negative field values separately,
    // then union the results. This is required to make sure stacks do not overlap. Negative
    // values stack in the negative direction and positive values stack in the positive
    // direction.
    let dataframe_pos = dataframe.filter(numeric_field.clone().gt_eq(lit(0)))?;
    let dataframe_pos = dataframe_pos.select(selection_0.clone())?;

    let dataframe_neg = dataframe.filter(numeric_field.clone().lt(lit(0)))?;
    let dataframe_neg = dataframe_neg.select(selection_0.clone())?;

    let dataframe = dataframe_pos.union(dataframe_neg)?;

    // Restore original order
    let dataframe = dataframe.sort(vec![Expr::Sort {
        expr: Box::new(col("__row_number")),
        asc: true,
        nulls_first: false,
    }])?;

    // Build selection_1
    let mut selection_1: Vec<_> = input_fields
        .iter()
        .filter_map(|field| {
            if field == alias0 || field == alias1 {
                None
            } else {
                Some(col(field))
            }
        })
        .collect();

    // Now compute alias1 column by adding numeric field to alias0
    let alias0_col = col(alias1).sub(numeric_field).alias(alias0);
    selection_1.push(alias0_col);
    selection_1.push(col(alias1));

    let dataframe = dataframe.select(selection_1.clone())?;
    Ok(dataframe)
}
