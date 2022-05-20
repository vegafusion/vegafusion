/*
 * VegaFusion
 * Copyright (C) 2022 VegaFusion Technologies LLC
 *
 * This program is distributed under multiple licenses.
 * Please consult the license documentation provided alongside
 * this program the details of the active license.
 */
use std::collections::{HashMap, HashSet};
use crate::proto::gen::tasks::Variable;
use crate::task_graph::graph::ScopedVariable;
use crate::task_graph::scope::TaskScope;

pub type VlSelectionFields = HashMap<String, Vec<String>>;
pub type VlSelectionFields2 = HashMap<ScopedVariable, Vec<String>>;

/// Enum storing info on which dataset columns are used in a given context.
/// Due to the dynamic nature of Vega specifications, it's not always possible to statically
/// determine which columns from a dataset will be used at runtime. In this case the
/// ColumnUsage::Unknown variant is used.  In the context of projection pushdown,
/// the ColumnUsage::Unknown variant indicates that all of original dataset columns must be
/// maintained
#[derive(Clone, Debug, PartialEq)]
pub enum ColumnUsage {
    Unknown,
    Known(HashSet<String>),
}

impl ColumnUsage {
    pub fn empty() -> ColumnUsage {
        ColumnUsage::Known(Default::default())
    }

    pub fn with_column(&self, column: &str) -> ColumnUsage {
        self.union(&ColumnUsage::from(vec![column].as_slice()))
    }

    /// Take the union of two ColumnUsage instances. If both are ColumnUsage::Known, then take
    /// the union of their known columns. If either is ColumnUsage::Unknown, then the union is
    /// also Unknown.
    pub fn union(&self, other: &ColumnUsage) -> ColumnUsage {
        match (self, other) {
            (ColumnUsage::Known(self_cols), ColumnUsage::Known(other_cols)) => {
                // If both column usages are known, we can union the known columns
                let new_cols: HashSet<_> = self_cols.union(other_cols).cloned().collect();
                ColumnUsage::Known(new_cols)
            }
            _ => {
                // If either is Unknown, then the union is unknown
                ColumnUsage::Unknown
            }
        }
    }
}

impl From<&[&str]> for ColumnUsage {
    fn from(columns: &[&str]) -> Self {
        let columns: HashSet<_> = columns.iter().map(|s| s.to_string()).collect();
        Self::Known(columns)
    }
}

impl From<&[String]> for ColumnUsage {
    fn from(columns: &[String]) -> Self {
        let columns: HashSet<_> = columns.iter().cloned().collect();
        Self::Known(columns)
    }
}


pub trait GetColumnUsage {
    fn column_usage(&self, vl_selection_fields: &VlSelectionFields) -> ColumnUsage;
}


/// Struct that tracks the usage of all columns across a collection of datasets
pub struct DatasetsColumnUsage {
    usages: HashMap<ScopedVariable, ColumnUsage>
}

impl DatasetsColumnUsage {
    pub fn empty() -> Self {
        Self {
            usages: Default::default()
        }
    }

    /// Take the union of two DatasetColumnUsage instances.
    pub fn union(&self, other: &DatasetsColumnUsage) -> DatasetsColumnUsage {
        let self_vars: HashSet<_> = self.usages.keys().cloned().collect();
        let other_vars: HashSet<_> = other.usages.keys().cloned().collect();
        let union_vars: HashSet<_> = self_vars.union(&other_vars).cloned().collect();

        let mut usages: HashMap<ScopedVariable, ColumnUsage> = HashMap::new();
        for var in union_vars {
            let self_usage = self.usages.get(&var).cloned().unwrap_or_else(|| ColumnUsage::empty());
            let other_usage = other.usages.get(&var).cloned().unwrap_or_else(|| ColumnUsage::empty());
            let combined_usage = self_usage.union(&other_usage);
            usages[var] = combined_usage;
        }

        Self {
            usages
        }
    }
}

pub trait GetDatasetColumnUsage {
    fn dataset_column_usage(
        &self,
        datum_name: &str,
        usage_scope: &[u32],
        task_scope: &TaskScope,
        vl_selection_fields: &VlSelectionFields2
    ) -> DatasetColumnUsage;
}


#[cfg(test)]
mod tests {
    use crate::expression::column_usage::ColumnUsage;

    #[test]
    fn test_with_column() {
        let left = ColumnUsage::from(vec!["one", "two"].as_slice());
        let result = left.with_column("three").with_column("four");
        let expected = ColumnUsage::from(vec!["one", "two", "three", "four"].as_slice());
        assert_eq!(result, expected)
    }

    #[test]
    fn test_union_known_known() {
        let left = ColumnUsage::from(vec!["one", "two"].as_slice());
        let right = ColumnUsage::from(vec!["two", "three", "four"].as_slice());
        let union = left.union(&right);
        let expected = ColumnUsage::from(vec!["one", "two", "three", "four"].as_slice());
        assert_eq!(union, expected)
    }

    #[test]
    fn test_union_known_unknown() {
        let left = ColumnUsage::from(vec!["one", "two"].as_slice());
        let union = left.union(&ColumnUsage::Unknown);
        assert_eq!(union, ColumnUsage::Unknown)
    }

    #[test]
    fn test_union_unknown_known() {
        let right = ColumnUsage::from(vec!["two", "three", "four"].as_slice());
        let union = ColumnUsage::Unknown.union(&right);
        assert_eq!(union, ColumnUsage::Unknown)
    }

    #[test]
    fn test_union_unknown_unknown() {
        let union = ColumnUsage::Unknown.union(&ColumnUsage::Unknown);
        assert_eq!(union, ColumnUsage::Unknown)
    }
}