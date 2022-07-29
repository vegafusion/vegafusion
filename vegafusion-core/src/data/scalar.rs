/*
 * VegaFusion
 * Copyright (C) 2022 VegaFusion Technologies LLC
 *
 * This program is distributed under multiple licenses.
 * Please consult the license documentation provided alongside
 * this program the details of the active license.
 */
use crate::arrow::datatypes::DataType;
use crate::error::{Result, VegaFusionError};
pub use datafusion_common::ScalarValue;

use arrow::datatypes::Field;
use serde_json::{Map, Value};
use std::convert::TryFrom;
use std::ops::Deref;

// Prefix for special values JSON encoded as strings
pub const DATETIME_PREFIX: &str = "__$datetime:";

pub trait ScalarValueHelpers {
    fn from_json(value: &Value) -> Result<ScalarValue>;
    fn to_json(&self) -> Result<Value>;

    fn to_f64(&self) -> Result<f64>;
    fn to_f64x2(&self) -> Result<[f64; 2]>;
    fn to_scalar_string(&self) -> Result<String>;
}

impl ScalarValueHelpers for ScalarValue {
    fn from_json(value: &Value) -> Result<ScalarValue> {
        let scalar_value = match value {
            Value::Null => {
                // Use None float64 for null
                ScalarValue::try_from(&DataType::Float64).unwrap()
            }
            Value::Bool(v) => ScalarValue::from(*v),
            Value::Number(v) => ScalarValue::from(v.as_f64().unwrap()),
            Value::String(v) => {
                if v.starts_with(DATETIME_PREFIX) {
                    let ms: i64 = v.strip_prefix(DATETIME_PREFIX).unwrap().parse().unwrap();
                    ScalarValue::Float64(Some(ms as f64))
                } else {
                    ScalarValue::from(v.as_str())
                }
            }
            Value::Object(values) => {
                let mut values: Vec<_> = values
                    .iter()
                    .map(|(name, val)| Ok((name.as_str(), ScalarValue::from_json(val)?)))
                    .collect::<Result<Vec<(&str, ScalarValue)>>>()?;

                // Sort keys for stability
                values.sort_by_key(|el| el.0);

                if values.is_empty() {
                    ScalarValue::from(vec![(
                        "__dummy",
                        ScalarValue::try_from(&DataType::Float64).unwrap(),
                    )])
                } else {
                    ScalarValue::from(values)
                }
            }
            Value::Array(elements) => {
                let (elements, dtype) = if elements.is_empty() {
                    (Vec::new(), DataType::Float64)
                } else {
                    let elements: Vec<_> = elements
                        .iter()
                        .map(ScalarValue::from_json)
                        .collect::<Result<Vec<ScalarValue>>>()?;
                    let dtype = elements[0].get_datatype();
                    (elements, dtype)
                };

                ScalarValue::List(Some(elements), Box::new(Field::new("item", dtype, true)))
            }
        };
        Ok(scalar_value)
    }

    fn to_json(&self) -> Result<Value> {
        let res = match self {
            ScalarValue::Boolean(Some(v)) => Value::from(*v),
            ScalarValue::Float32(Some(v)) => Value::from(*v),
            ScalarValue::Float64(Some(v)) => Value::from(*v),
            ScalarValue::Int8(Some(v)) => Value::from(*v),
            ScalarValue::Int16(Some(v)) => Value::from(*v),
            ScalarValue::Int32(Some(v)) => Value::from(*v),
            ScalarValue::Int64(Some(v)) => Value::from(*v),
            ScalarValue::UInt8(Some(v)) => Value::from(*v),
            ScalarValue::UInt16(Some(v)) => Value::from(*v),
            ScalarValue::UInt32(Some(v)) => Value::from(*v),
            ScalarValue::UInt64(Some(v)) => Value::from(*v),
            ScalarValue::Utf8(Some(v)) => Value::from(v.clone()),
            ScalarValue::LargeUtf8(Some(v)) => Value::from(v.clone()),
            ScalarValue::Binary(Some(_v)) => {
                unimplemented!()
            }
            ScalarValue::LargeBinary(Some(_v)) => {
                unimplemented!()
            }
            ScalarValue::Date32(Some(v)) => {
                let ms_per_day: i32 = 1000 * 60 * 60 * 24;
                let utc_millis = *v * ms_per_day;
                Value::from(utc_millis)
            }
            ScalarValue::Date64(Some(v)) => {
                // To UTC integer milliseconds (alread in UTC)
                Value::from(*v)
            }
            ScalarValue::TimestampSecond(Some(_v), _) => {
                unimplemented!()
            }
            ScalarValue::TimestampMillisecond(Some(_v), _) => {
                unimplemented!()
            }
            ScalarValue::TimestampMicrosecond(Some(_v), _) => {
                unimplemented!()
            }
            ScalarValue::TimestampNanosecond(Some(_v), _) => {
                unimplemented!()
            }
            ScalarValue::IntervalYearMonth(Some(_v)) => {
                unimplemented!()
            }
            ScalarValue::IntervalDayTime(Some(_v)) => {
                unimplemented!()
            }
            ScalarValue::List(Some(v), _) => Value::Array(
                v.clone()
                    .into_iter()
                    .map(|v| v.to_json())
                    .collect::<Result<Vec<_>>>()?,
            ),
            ScalarValue::Struct(Some(v), fields) => {
                let mut pairs: Map<String, Value> = Default::default();
                for (val, field) in v.iter().zip(fields.deref()) {
                    pairs.insert(field.name().clone(), val.to_json()?);
                }
                Value::Object(pairs)
            }
            _ => Value::Null,
        };

        Ok(res)
    }

    fn to_f64(&self) -> Result<f64> {
        Ok(match self {
            ScalarValue::Float32(Some(e)) => *e as f64,
            ScalarValue::Float64(Some(e)) => *e,
            ScalarValue::Int8(Some(e)) => *e as f64,
            ScalarValue::Int16(Some(e)) => *e as f64,
            ScalarValue::Int32(Some(e)) => *e as f64,
            ScalarValue::Int64(Some(e)) => *e as f64,
            ScalarValue::UInt8(Some(e)) => *e as f64,
            ScalarValue::UInt16(Some(e)) => *e as f64,
            ScalarValue::UInt32(Some(e)) => *e as f64,
            ScalarValue::UInt64(Some(e)) => *e as f64,
            _ => {
                return Err(VegaFusionError::internal(&format!(
                    "Cannot convert {} to f64",
                    self
                )))
            }
        })
    }

    fn to_f64x2(&self) -> Result<[f64; 2]> {
        if let ScalarValue::List(Some(elements), _) = self {
            if let [v0, v1] = elements.as_slice() {
                return Ok([v0.to_f64()?, v1.to_f64()?]);
            }
        }
        Err(VegaFusionError::internal(&format!(
            "Cannot convert {} to [f64; 2]",
            self
        )))
    }

    fn to_scalar_string(&self) -> Result<String> {
        Ok(match self {
            ScalarValue::Utf8(Some(value)) => value.clone(),
            ScalarValue::LargeUtf8(Some(value)) => value.clone(),
            _ => {
                return Err(VegaFusionError::internal(&format!(
                    "Cannot convert {} to String",
                    self
                )))
            }
        })
    }
}
