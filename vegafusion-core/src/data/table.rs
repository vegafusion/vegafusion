/*
 * VegaFusion
 * Copyright (C) 2022 VegaFusion Technologies LLC
 *
 * This program is distributed under multiple licenses.
 * Please consult the license documentation provided alongside
 * this program the details of the active license.
 */
use crate::arrow::{
    datatypes::{DataType, SchemaRef},
    json,
    record_batch::RecordBatch,
};
use crate::error::{Result, ResultWithContext, VegaFusionError};
use std::borrow::Cow;

use std::convert::TryFrom;
use std::sync::Arc;

use crate::arrow::ipc::reader::StreamReader;
use crate::arrow::ipc::writer::StreamWriter;
use std::hash::{Hash, Hasher};
use std::io::Cursor;

use super::scalar::ScalarValue;
use crate::arrow::array::ArrayRef;
use crate::data::json_writer::record_batches_to_json_rows;

use arrow::array::StructArray;
use arrow::datatypes::Field;
use arrow::json::reader::DecoderOptions;
use serde_json::{json, Value};

#[derive(Clone, Debug)]
pub struct VegaFusionTable {
    pub schema: SchemaRef,
    pub batches: Vec<RecordBatch>,
}

impl VegaFusionTable {
    pub fn try_new(schema: SchemaRef, partitions: Vec<RecordBatch>) -> Result<Self> {
        if partitions
            .iter()
            .all(|batches| schema.contains(&batches.schema()))
        {
            Ok(Self {
                schema,
                batches: partitions,
            })
        } else {
            Err(VegaFusionError::internal(
                "Mismatch between schema and batches",
            ))
        }
    }

    pub fn num_rows(&self) -> usize {
        self.batches.iter().map(|batch| batch.num_rows()).sum()
    }

    /// Keep, at most, the first n rows
    pub fn head(&self, n: usize) -> Self {
        let mut so_far = 0;
        let mut head_batches: Vec<RecordBatch> = Vec::new();

        for batch in &self.batches {
            if so_far == n {
                break;
            } else if so_far + batch.num_rows() <= n {
                // Keep full batch
                so_far += batch.num_rows();
                head_batches.push(batch.clone());
            } else {
                // Keep partial batch
                let keep = n - so_far;
                head_batches.push(batch.slice(0, keep));
                break;
            }
        }

        Self {
            schema: self.schema.clone(),
            batches: head_batches,
        }
    }

    pub fn batches(&self) -> &Vec<RecordBatch> {
        &self.batches
    }

    pub fn to_record_batch(&self) -> Result<RecordBatch> {
        let mut schema = self.schema.clone();
        if let Some(batch) = self.batches.get(0) {
            schema = batch.schema()
        }
        RecordBatch::concat(&schema, &self.batches)
            .with_context(|| String::from("Failed to concatenate RecordBatches"))
    }

    pub fn to_scalar_value(&self) -> Result<ScalarValue> {
        if self.num_rows() == 0 {
            // Return empty list with (arbitrary) Float64 type
            let dtype = DataType::Float64;
            return Ok(ScalarValue::List(
                Some(Vec::new()),
                Box::new(Field::new("item", dtype, true)),
            ));
        }

        let mut elements: Vec<ScalarValue> = Vec::new();
        for batch in &self.batches {
            let array = Arc::new(StructArray::from(batch.clone())) as ArrayRef;

            for i in 0..array.len() {
                let scalar = ScalarValue::try_from_array(&array, i).with_context(|| {
                    "Failed to convert record batch row to ScalarValue".to_string()
                })?;
                elements.push(scalar)
            }
        }

        let dtype = elements[0].get_datatype();
        Ok(ScalarValue::List(
            Some(elements),
            Box::new(Field::new("item", dtype, true)),
        ))
    }

    pub fn to_json(&self) -> serde_json::Value {
        let mut rows: Vec<serde_json::Value> = Vec::with_capacity(self.num_rows());
        for row in record_batches_to_json_rows(&self.batches) {
            rows.push(serde_json::Value::Object(row));
        }
        serde_json::Value::Array(rows)
    }

    pub fn from_json(value: &serde_json::Value, batch_size: usize) -> Result<Self> {
        if let serde_json::Value::Array(values) = value {
            // Handle special case where array elements are non-object scalars
            let mut values = Cow::Borrowed(values);
            if let Some(first) = values.get(0) {
                if let Value::Object(props) = first {
                    // Handle odd special case where vega will interpret
                    // [{}, {}] as [{"datum": {}}, {"datum": {}}]
                    if props.is_empty() {
                        values = Cow::Owned(vec![json!({"datum": {"__dummy": 0}}); values.len()])
                    }
                } else {
                    // Array of scalars, need to wrap elements objects with "data" field
                    values = Cow::Owned(
                        values
                            .iter()
                            .map(|value| json!({ "data": value }))
                            .collect(),
                    )
                }
            }

            let schema_result = json::reader::infer_json_schema_from_iterator(
                values.iter().take(1024).map(|v| Ok(v.clone())),
            );

            match schema_result {
                Err(_) => {
                    // This happens when array elements are objects with no fields
                    let empty_scalar = ScalarValue::from(vec![(
                        "__dummy",
                        ScalarValue::try_from(&DataType::Float64).unwrap(),
                    )]);
                    let array = empty_scalar.to_array_of_size(values.len());
                    let struct_array = array.as_any().downcast_ref::<StructArray>().unwrap();
                    let record_batch = RecordBatch::from(struct_array);
                    Self::try_new(record_batch.schema(), vec![record_batch])
                }
                Ok(schema) => {
                    let schema_ref = Arc::new(schema);

                    // read record batches
                    let decoder = json::reader::Decoder::new(
                        schema_ref.clone(),
                        DecoderOptions::default().with_batch_size(batch_size),
                    );
                    let mut batches: Vec<RecordBatch> = Vec::new();
                    let mut value_iter = values.iter().map(|v| Ok(v.clone()));

                    while let Some(batch) = decoder
                        .next_batch(&mut value_iter)
                        .with_context(|| "Failed to read json to arrow")?
                    {
                        batches.push(batch);
                    }

                    Self::try_new(schema_ref, batches)
                }
            }
        } else {
            Err(VegaFusionError::internal(&format!(
                "Expected JSON array, not: {}",
                value
            )))
        }
    }

    // Serialize to bytes using Arrow IPC format
    pub fn to_ipc_bytes(&self) -> Result<Vec<u8>> {
        let buffer: Vec<u8> = Vec::new();
        let mut stream_writer = StreamWriter::try_new(buffer, self.schema.as_ref())?;

        for batch in &self.batches {
            stream_writer.write(batch)?;
        }

        stream_writer.finish()?;
        Ok(stream_writer.into_inner()?)
    }

    pub fn from_ipc_bytes(data: &[u8]) -> Result<Self> {
        let cursor = Cursor::new(data);
        let reader = StreamReader::try_new(cursor, None)?;
        let schema = reader.schema();
        let mut batches: Vec<RecordBatch> = Vec::new();

        for batch in reader {
            batches.push(batch?);
        }

        Ok(Self { schema, batches })
    }
}

impl From<RecordBatch> for VegaFusionTable {
    fn from(value: RecordBatch) -> Self {
        Self {
            schema: value.schema(),
            batches: vec![value],
        }
    }
}

impl Hash for VegaFusionTable {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.to_ipc_bytes().unwrap().hash(state)
    }
}
