/*
 * VegaFusion
 * Copyright (C) 2022 Jon Mease
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public
 * License along with this program.
 * If not, see http://www.gnu.org/licenses/.
 */
use crate::data::scalar::{ScalarValue, ScalarValueHelpers};
use crate::data::table::VegaFusionTable;
use crate::error::Result;
use crate::error::VegaFusionError;
use crate::planning::stitch::CommPlan;
use crate::proto::gen::tasks::{Variable, VariableNamespace};
use crate::task_graph::graph::ScopedVariable;
use crate::task_graph::task_value::TaskValue;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::convert::TryFrom;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WatchNamespace {
    Signal,
    Data,
}

impl TryFrom<VariableNamespace> for WatchNamespace {
    type Error = VegaFusionError;

    fn try_from(value: VariableNamespace) -> Result<Self> {
        match value {
            VariableNamespace::Signal => Ok(Self::Signal),
            VariableNamespace::Data => Ok(Self::Data),
            _ => Err(VegaFusionError::internal("Scale namespace not supported")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Watch {
    pub namespace: WatchNamespace,
    pub name: String,
    pub scope: Vec<u32>,
}

impl Watch {
    pub fn to_scoped_variable(&self) -> ScopedVariable {
        (
            match self.namespace {
                WatchNamespace::Signal => Variable::new_signal(&self.name),
                WatchNamespace::Data => Variable::new_data(&self.name),
            },
            self.scope.clone(),
        )
    }
}

impl TryFrom<ScopedVariable> for Watch {
    type Error = VegaFusionError;

    fn try_from(value: ScopedVariable) -> Result<Self> {
        let tmp = value.0.namespace();
        let tmp = WatchNamespace::try_from(tmp)?;
        Ok(Self {
            namespace: tmp,
            name: value.0.name.clone(),
            scope: value.1,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WatchPlan {
    pub server_to_client: Vec<Watch>,
    pub client_to_server: Vec<Watch>,
}

impl From<CommPlan> for WatchPlan {
    fn from(value: CommPlan) -> Self {
        Self {
            server_to_client: value
                .server_to_client
                .into_iter()
                .map(|scoped_var| Watch::try_from(scoped_var).unwrap())
                .sorted()
                .collect(),
            client_to_server: value
                .client_to_server
                .into_iter()
                .map(|scoped_var| Watch::try_from(scoped_var).unwrap())
                .sorted()
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WatchValue {
    pub watch: Watch,
    pub value: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WatchValues {
    pub values: Vec<WatchValue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExportUpdateNamespace {
    Signal,
    Data,
}

impl TryFrom<VariableNamespace> for ExportUpdateNamespace {
    type Error = VegaFusionError;

    fn try_from(value: VariableNamespace) -> Result<Self> {
        match value {
            VariableNamespace::Signal => Ok(Self::Signal),
            VariableNamespace::Data => Ok(Self::Data),
            _ => Err(VegaFusionError::internal("Scale namespace not supported")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportUpdate {
    pub namespace: ExportUpdateNamespace,
    pub name: String,
    pub scope: Vec<u32>,
    pub value: Value,
}

impl ExportUpdate {
    pub fn to_scoped_var(&self) -> ScopedVariable {
        let namespace = match self.namespace {
            ExportUpdateNamespace::Signal => VariableNamespace::Signal as i32,
            ExportUpdateNamespace::Data => VariableNamespace::Data as i32,
        };

        (
            Variable {
                name: self.name.clone(),
                namespace: namespace,
            },
            self.scope.clone(),
        )
    }

    pub fn to_task_value(&self) -> TaskValue {
        match self.namespace {
            ExportUpdateNamespace::Signal => {
                TaskValue::Scalar(ScalarValue::from_json(&self.value).unwrap())
            }
            ExportUpdateNamespace::Data => {
                TaskValue::Table(VegaFusionTable::from_json(&self.value, 1024).unwrap())
            }
        }
    }
}

pub type ExportUpdateBatch = Vec<ExportUpdate>;
