/*
 * VegaFusion
 * Copyright (C) 2022 VegaFusion Technologies LLC
 *
 * This program is distributed under multiple licenses.
 * Please consult the license documentation provided alongside
 * this program the details of the active license.
 */
use crate::expression::compiler::builtin_functions::date_time::date_parsing::{
    make_date_str_to_millis_udf, DateParseMode,
};
use crate::expression::compiler::utils::{cast_to, is_string_datatype};
use crate::task_graph::timezone::RuntimeTzConfig;
use chrono::{DateTime, TimeZone};
use datafusion::arrow::array::{Array, ArrayRef, Int64Array};
use datafusion::arrow::datatypes::DataType;
use datafusion::logical_plan::{DFSchema, Expr, ExprSchemable};
use datafusion::physical_plan::udf::ScalarUDF;
use datafusion::physical_plan::ColumnarValue;
use datafusion::scalar::ScalarValue;
use datafusion_expr::{
    ReturnTypeFunction, ScalarFunctionImplementation, Signature, TypeSignature, Volatility,
};
use std::str::FromStr;
use std::sync::Arc;
use vegafusion_core::error::{Result, ResultWithContext, VegaFusionError};

pub fn to_date_transform(
    tz_config: &RuntimeTzConfig,
    args: &[Expr],
    schema: &DFSchema,
) -> Result<Expr> {
    // Datetime from string or integer in milliseconds
    let mut arg = args[0].clone();
    let dtype = arg
        .get_type(schema)
        .with_context(|| format!("Failed to infer type of expression: {:?}", arg))?;

    if is_string_datatype(&dtype) {
        let default_input_tz = if args.len() == 2 {
            // Second argument is a an override local timezone string
            let input_tz_expr = &args[1];
            if let Expr::Literal(ScalarValue::Utf8(Some(input_tz_str))) = input_tz_expr {
                if input_tz_str == "local" {
                    tz_config.local_tz
                } else {
                    chrono_tz::Tz::from_str(input_tz_str)
                        .ok()
                        .with_context(|| {
                            format!("Failed to parse {} as a timezone", input_tz_str)
                        })?
                }
            } else {
                return Err(VegaFusionError::parse(
                    "Second argument to toDate must be a timezone string",
                ));
            }
        } else {
            tz_config.default_input_tz
        };

        arg = Expr::ScalarUDF {
            fun: Arc::new(make_date_str_to_millis_udf(
                DateParseMode::JavaScript,
                &Some(default_input_tz),
            )),
            args: vec![arg],
        }
    }

    cast_to(arg, &DataType::Int64, schema)
}

pub fn datetime_transform(
    tz_config: &RuntimeTzConfig,
    args: &[Expr],
    schema: &DFSchema,
) -> Result<Expr> {
    if args.len() == 1 {
        // Datetime from string or integer in milliseconds
        let mut arg = args[0].clone();
        let dtype = arg
            .get_type(schema)
            .with_context(|| format!("Failed to infer type of expression: {:?}", arg))?;

        if is_string_datatype(&dtype) {
            arg = Expr::ScalarUDF {
                fun: Arc::new(make_date_str_to_millis_udf(
                    DateParseMode::JavaScript,
                    &Some(tz_config.default_input_tz),
                )),
                args: vec![arg],
            }
        }

        cast_to(arg, &DataType::Int64, schema)
    } else {
        // Numeric date components
        let int_args: Vec<_> = args
            .iter()
            .map(|arg| cast_to(arg.clone(), &DataType::Int64, schema))
            .collect::<Result<Vec<_>>>()?;

        Ok(Expr::ScalarUDF {
            fun: Arc::new(make_datetime_components_udf(tz_config.default_input_tz)),
            args: int_args,
        })
    }
}

pub fn make_datetime_components_udf(input_tz: chrono_tz::Tz) -> ScalarUDF {
    let datetime_components: ScalarFunctionImplementation =
        Arc::new(move |args: &[ColumnarValue]| {
            // pad with defaults out to 7 arguments
            let num_args = args.len();
            let mut args = Vec::from(args);

            if args.len() < 2 {
                // default to 1st month of the year
                args.push(ColumnarValue::Scalar(ScalarValue::Int64(Some(0))));
            }

            if args.len() < 3 {
                // default to 1st of the month
                args.push(ColumnarValue::Scalar(ScalarValue::Int64(Some(1))));
            }

            // Remaining args (hour, minute, second, millisecond) default to zero
            for _ in num_args..7 {
                args.push(ColumnarValue::Scalar(ScalarValue::Int64(Some(0))));
            }

            // first, identify if any of the arguments is an Array. If yes, store its `len`,
            // as any scalar will need to be converted to an array of len `len`.
            let len = args
                .iter()
                .fold(Option::<usize>::None, |acc, arg| match arg {
                    ColumnarValue::Scalar(_) => acc,
                    ColumnarValue::Array(a) => Some(a.len()),
                });

            // to arrays
            let args = if let Some(len) = len {
                args.iter()
                    .map(|arg| arg.clone().into_array(len))
                    .collect::<Vec<ArrayRef>>()
            } else {
                args.iter()
                    .map(|arg| arg.clone().into_array(1))
                    .collect::<Vec<ArrayRef>>()
            };

            // To int64 arrays
            let years = args[0].as_any().downcast_ref::<Int64Array>().unwrap();
            let months = args[1].as_any().downcast_ref::<Int64Array>().unwrap();
            let days = args[2].as_any().downcast_ref::<Int64Array>().unwrap();
            let hours = args[3].as_any().downcast_ref::<Int64Array>().unwrap();
            let minutes = args[4].as_any().downcast_ref::<Int64Array>().unwrap();
            let seconds = args[5].as_any().downcast_ref::<Int64Array>().unwrap();
            let milliseconds = args[6].as_any().downcast_ref::<Int64Array>().unwrap();

            let num_rows = years.len();
            let mut datetime_builder = Int64Array::builder(num_rows);

            for i in 0..num_rows {
                if years.is_null(i)
                    || months.is_null(i)
                    || days.is_null(i)
                    || hours.is_null(i)
                    || minutes.is_null(i)
                    || seconds.is_null(i)
                    || milliseconds.is_null(i)
                {
                    // If any component is null, propagate null
                    datetime_builder.append_null();
                } else {
                    let year = years.value(i);
                    let month = months.value(i);
                    let day = days.value(i);
                    let hour = hours.value(i);
                    let minute = minutes.value(i);
                    let second = seconds.value(i);
                    let millisecond = milliseconds.value(i);

                    // Treat 00-99 as 1900 to 1999
                    let mut year = year;
                    if (0..100).contains(&year) {
                        year += 1900
                    }

                    let datetime: Option<DateTime<_>> = input_tz
                        .ymd_opt(year as i32, month as u32 + 1, day as u32)
                        .single()
                        .and_then(|date| {
                            date.and_hms_milli_opt(
                                hour as u32,
                                minute as u32,
                                second as u32,
                                millisecond as u32,
                            )
                        });

                    if let Some(datetime) = datetime {
                        // Always convert to UTC
                        let datetime = datetime.with_timezone(&chrono::Utc);
                        datetime_builder.append_value(datetime.timestamp_millis());
                    } else {
                        // Invalid date
                        datetime_builder.append_null();
                    }
                }
            }

            let result = Arc::new(datetime_builder.finish()) as ArrayRef;

            // maybe back to scalar
            if len.is_some() {
                Ok(ColumnarValue::Array(result))
            } else {
                ScalarValue::try_from_array(&result, 0).map(ColumnarValue::Scalar)
            }
        });

    let return_type: ReturnTypeFunction = Arc::new(move |_| Ok(Arc::new(DataType::Int64)));

    // vega signature: datetime(year, month[, day, hour, min, sec, millisec])
    let sig = |n: usize| vec![DataType::Int64; n];
    let signature = Signature::one_of(
        vec![
            TypeSignature::Exact(sig(1)),
            TypeSignature::Exact(sig(2)),
            TypeSignature::Exact(sig(3)),
            TypeSignature::Exact(sig(4)),
            TypeSignature::Exact(sig(5)),
            TypeSignature::Exact(sig(6)),
            TypeSignature::Exact(sig(7)),
        ],
        Volatility::Immutable,
    );
    ScalarUDF::new(
        "utc_datetime_components",
        &signature,
        &return_type,
        &datetime_components,
    )
}

lazy_static! {
    pub static ref UTC_COMPONENTS: ScalarUDF = make_datetime_components_udf(chrono_tz::UTC);
}
