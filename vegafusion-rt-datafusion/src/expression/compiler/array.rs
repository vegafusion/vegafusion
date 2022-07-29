/*
 * VegaFusion
 * Copyright (C) 2022 VegaFusion Technologies LLC
 *
 * This program is distributed under multiple licenses.
 * Please consult the license documentation provided alongside
 * this program the details of the active license.
 */
use crate::expression::compiler::{compile, config::CompilationConfig};
use datafusion::arrow::array::{
    Array, ArrayDataBuilder, ArrayRef, BooleanBufferBuilder, Int32Array, ListArray,
};
use datafusion::arrow::datatypes::{DataType, Field, Float64Type};
use datafusion::logical_plan::{DFSchema, Expr};
use datafusion::physical_plan::functions::make_scalar_function;
use datafusion::physical_plan::udf::ScalarUDF;
use datafusion_expr::{ReturnTypeFunction, Signature, Volatility};
use std::ops::Deref;
use std::sync::Arc;
use vegafusion_core::error::Result;
use vegafusion_core::proto::gen::expression::ArrayExpression;

lazy_static! {
    pub static ref ARRAY_CONSTRUCTOR_UDF: ScalarUDF = make_array_constructor_udf();
}

pub fn array_constructor_udf() -> ScalarUDF {
    ARRAY_CONSTRUCTOR_UDF.deref().clone()
}

pub fn compile_array(
    node: &ArrayExpression,
    config: &CompilationConfig,
    schema: &DFSchema,
) -> Result<Expr> {
    let mut elements: Vec<Expr> = Vec::new();
    for el in &node.elements {
        let phys_expr = compile(el, config, Some(schema))?;
        elements.push(phys_expr);
    }
    Ok(Expr::ScalarUDF {
        fun: Arc::new(array_constructor_udf()),
        args: elements,
    })
}

pub fn make_array_constructor_udf() -> ScalarUDF {
    let array_constructor = |args: &[ArrayRef]| {
        // Signature ensures arguments have the same type and length
        let num_args = args.len();

        // Return empty array as single row, empty array with Float64 data type
        if num_args == 0 {
            let empty_row = vec![Some(Vec::<Option<f64>>::new())];
            let empty = ListArray::from_iter_primitive::<Float64Type, _, _>(empty_row);
            return Ok(Arc::new(empty) as ArrayRef);
        }

        let num_rows = args[0].len();
        let element_dtype = args[0].data_type();
        let array_dtype = DataType::List(Box::new(Field::new("item", element_dtype.clone(), true)));

        // Concatenate arrays into single flat array (in column-major ordering)
        let arrays: Vec<&dyn Array> = args.iter().map(|a| a.as_ref()).collect();
        let concatted = datafusion::arrow::compute::concat(&arrays).unwrap();

        // Transpose array into row-major ordering and compute offset indices
        let mut indices_builder = Int32Array::builder(num_rows * num_args);
        let mut flat_valid_builder = BooleanBufferBuilder::new(num_rows * num_args);
        let mut offsets_builder = Int32Array::builder(num_rows + 1);

        offsets_builder.append_value(0);

        for r in 0..num_rows {
            for (a, arg) in args.iter().enumerate() {
                let col_major_idx = (a * num_rows + r) as i32;
                indices_builder.append_value(col_major_idx);
                flat_valid_builder.append(arg.is_valid(r));
            }
            offsets_builder.append_value((num_args * (r + 1)) as i32);
        }

        let flat_values = datafusion::arrow::compute::take(
            concatted.as_ref(),
            &indices_builder.finish(),
            Default::default(),
        )
        .unwrap();
        let offsets = offsets_builder.finish();

        // Build ListArray data
        let list_array_data = ArrayDataBuilder::new(array_dtype)
            .len(num_rows)
            .null_bit_buffer(Some(flat_valid_builder.finish()))
            .add_buffer(offsets.data().buffers()[0].clone())
            .add_child_data(flat_values.data().clone())
            .build()?;

        Ok(Arc::new(ListArray::from(list_array_data)) as ArrayRef)
    };
    let array_constructor = make_scalar_function(array_constructor);

    let return_type: ReturnTypeFunction = Arc::new(move |args| {
        Ok(Arc::new(DataType::List(Box::new(Field::new(
            "item",
            if args.is_empty() {
                DataType::Float64
            } else {
                args[0].clone()
            },
            true,
        )))))
    });
    ScalarUDF::new(
        "list",
        &Signature::variadic_equal(Volatility::Immutable),
        &return_type,
        &array_constructor,
    )
}
