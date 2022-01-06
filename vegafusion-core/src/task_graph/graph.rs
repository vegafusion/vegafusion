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
use crate::error::{Result, ResultWithContext, VegaFusionError};
use crate::proto::gen::tasks::{
    IncomingEdge, NodeValueIndex, OutgoingEdge, Task, TaskGraph, TaskNode, Variable,
};
use crate::task_graph::scope::TaskScope;
use petgraph::algo::toposort;
use petgraph::graph::NodeIndex;
use petgraph::prelude::EdgeRef;
use petgraph::Direction;
use std::collections::HashMap;

use crate::task_graph::task_value::TaskValue;

use crate::proto::gen::tasks::task::TaskKind;
use crate::proto::gen::tasks::task_value::Data;
use crate::proto::gen::tasks::TaskValue as ProtoTaskValue;
use std::collections::hash_map::DefaultHasher;
use std::convert::TryFrom;
use std::hash::{Hash, Hasher};

struct PetgraphEdge {
    output_var: Option<Variable>,
}

pub type ScopedVariable = (Variable, Vec<u32>);

impl TaskGraph {
    pub fn new(tasks: Vec<Task>, task_scope: &TaskScope) -> Result<Self> {
        let mut graph: petgraph::graph::DiGraph<ScopedVariable, PetgraphEdge> =
            petgraph::graph::DiGraph::new();
        let mut tasks_map: HashMap<ScopedVariable, (NodeIndex, Task)> = HashMap::new();

        // Add graph nodes
        for task in tasks {
            // Add scope variable
            let scoped_var = (task.variable().clone(), task.scope.clone());
            let node_index = graph.add_node(scoped_var.clone());
            tasks_map.insert(scoped_var, (node_index, task));
        }

        // Resolve and add edges
        for (node_index, task) in tasks_map.values() {
            let usage_scope = task.scope();
            let input_vars = task.input_vars();
            for input_var in input_vars {
                let resolved = task_scope.resolve_scope(&input_var.var, usage_scope)?;
                let input_scoped_var = (resolved.var.clone(), resolved.scope.clone());
                let (input_node_index, _) =
                    tasks_map.get(&input_scoped_var).with_context(|| {
                        format!(
                            "No variable {:?} with scope {:?}",
                            input_scoped_var.0, input_scoped_var.1
                        )
                    })?;

                // Add graph edge
                if input_node_index != node_index {
                    // If a task depends on information generated by the task,that will be handled
                    // internally to the task. So we avoid making a cycle
                    graph.add_edge(
                        *input_node_index,
                        *node_index,
                        PetgraphEdge {
                            output_var: resolved.output_var.clone(),
                        },
                    );
                }
            }
        }

        // Create mapping from toposorted node_index to the final linear node index
        let toposorted: Vec<NodeIndex> = match toposort(&graph, None) {
            Err(err) => {
                return Err(VegaFusionError::internal(&format!(
                    "failed to sort dependency graph topologically: {:?}",
                    err
                )))
            }
            Ok(toposorted) => toposorted,
        };

        let toposorted_node_indexes: HashMap<NodeIndex, usize> = toposorted
            .iter()
            .enumerate()
            .map(|(sorted_index, node_index)| (*node_index, sorted_index))
            .collect();

        // Create linear vec of TaskNodes, with edges as sorted index references to nodes
        let task_nodes = toposorted
            .iter()
            .map(|node_index| {
                let scoped_var = graph.node_weight(*node_index).unwrap();
                let (_, task) = tasks_map.get(scoped_var).unwrap();

                // Collect outgoing node indexes
                let outgoing_node_ids: Vec<_> = graph
                    .edges_directed(*node_index, Direction::Outgoing)
                    .map(|edge| edge.target())
                    .collect();

                let outgoing: Vec<_> = outgoing_node_ids
                    .iter()
                    .map(|node_index| {
                        let sorted_index = *toposorted_node_indexes.get(node_index).unwrap() as u32;
                        OutgoingEdge {
                            target: sorted_index,
                            propagate: true,
                        }
                    })
                    .collect();

                // Collect incoming node indexes
                let incoming_node_ids: Vec<_> = graph
                    .edges_directed(*node_index, Direction::Incoming)
                    .map(|edge| (edge.source(), &edge.weight().output_var))
                    .collect();

                // Sort incoming nodes to match order expected by the task
                let incoming_vars: HashMap<_, _> = incoming_node_ids
                    .iter()
                    .map(|(node_index, output_var)| {
                        let var = graph.node_weight(*node_index).unwrap().0.clone();
                        ((var, (*output_var).clone()), node_index)
                    })
                    .collect();

                let incoming: Vec<_> = task
                    .input_vars()
                    .iter()
                    .filter_map(|var| {
                        let resolved = task_scope
                            .resolve_scope(&var.var, scoped_var.1.as_slice())
                            .unwrap();
                        let output_var = resolved.output_var.clone();
                        let resolved = (resolved.var, resolved.output_var);

                        let node_index = *incoming_vars.get(&resolved)?;
                        let sorted_index = *toposorted_node_indexes.get(node_index).unwrap() as u32;

                        if let Some(output_var) = output_var {
                            let weight = graph.node_weight(*node_index).unwrap();
                            let (_, input_task) = tasks_map.get(weight).unwrap();

                            let output_index = match input_task
                                .output_vars()
                                .iter()
                                .position(|v| v == &output_var)
                            {
                                Some(output_index) => output_index,
                                None => {
                                    return Some(Err(VegaFusionError::internal(
                                        "Failed to find output variable",
                                    )))
                                }
                            };

                            Some(Ok(IncomingEdge {
                                source: sorted_index,
                                output: Some(output_index as u32),
                            }))
                        } else {
                            Some(Ok(IncomingEdge {
                                source: sorted_index,
                                output: None,
                            }))
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;

                Ok(TaskNode {
                    task: Some(task.clone()),
                    incoming,
                    outgoing,
                    id_fingerprint: 0,
                    state_fingerprint: 0,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut this = Self { nodes: task_nodes };

        this.init_identity_fingerprints()?;
        this.update_state_fingerprints()?;

        Ok(this)
    }

    pub fn build_mapping(&self) -> HashMap<ScopedVariable, NodeValueIndex> {
        let mut mapping: HashMap<ScopedVariable, NodeValueIndex> = Default::default();
        for (node_index, node) in self.nodes.iter().enumerate() {
            let task = node.task();
            let _scope = task.scope.clone();
            let scoped_var = (task.variable().clone(), task.scope.clone());
            mapping.insert(scoped_var, NodeValueIndex::new(node_index as u32, None));

            for (output_index, output_var) in task.output_vars().into_iter().enumerate() {
                let scope_output_var = (output_var, task.scope.clone());
                mapping.insert(
                    scope_output_var,
                    NodeValueIndex::new(node_index as u32, Some(output_index as u32)),
                );
            }
        }
        mapping
    }

    fn init_identity_fingerprints(&mut self) -> Result<()> {
        // Compute new identity fingerprints
        let mut id_fingerprints: Vec<u64> = Vec::with_capacity(self.nodes.len());
        for (i, node) in self.nodes.iter().enumerate() {
            let task = node.task();
            let mut hasher = deterministic_hash::DeterministicHasher::new(DefaultHasher::new());

            if let TaskKind::Value(value) = task.task_kind() {
                // Only hash the distinction between Scalar and Table, not the value itself.
                // The state fingerprint takes the value into account.
                task.variable().hash(&mut hasher);
                task.scope.hash(&mut hasher);
                match value.data.as_ref().unwrap() {
                    Data::Scalar(_) => "scalar".hash(&mut hasher),
                    Data::Table(_) => "data".hash(&mut hasher),
                }
            } else {
                // Include id_fingerprint of parents in the hash
                for parent_index in self.parent_indices(i)? {
                    id_fingerprints[parent_index].hash(&mut hasher);
                }

                // Include current task in hash
                task.hash(&mut hasher)
            }

            id_fingerprints.push(hasher.finish());
        }

        // Apply fingerprints
        self.nodes
            .iter_mut()
            .zip(id_fingerprints)
            .for_each(|(node, fingerprint)| {
                node.id_fingerprint = fingerprint;
            });

        Ok(())
    }

    /// Update state finger prints of nodes, and return indices of nodes that were updated
    pub fn update_state_fingerprints(&mut self) -> Result<Vec<usize>> {
        // Compute new identity fingerprints
        let mut state_fingerprints: Vec<u64> = Vec::with_capacity(self.nodes.len());
        for (i, node) in self.nodes.iter().enumerate() {
            let task = node.task();
            let mut hasher = deterministic_hash::DeterministicHasher::new(DefaultHasher::new());

            if matches!(task.task_kind(), TaskKind::Value(_)) {
                // Hash the task with inline TaskValue
                task.hash(&mut hasher);
            } else {
                // Include state fingerprint of parents in the hash
                for parent_index in self.parent_indices(i)? {
                    state_fingerprints[parent_index].hash(&mut hasher);
                }

                // Include id fingerprint of current task
                node.id_fingerprint.hash(&mut hasher);
            }

            state_fingerprints.push(hasher.finish());
        }

        // Apply fingerprints
        let updated: Vec<_> = self
            .nodes
            .iter_mut()
            .zip(state_fingerprints)
            .enumerate()
            .filter_map(|(node_index, (node, fingerprint))| {
                if node.state_fingerprint != fingerprint {
                    node.state_fingerprint = fingerprint;
                    Some(node_index)
                } else {
                    None
                }
            })
            .collect();

        Ok(updated)
    }

    pub fn update_value(
        &mut self,
        node_index: usize,
        value: TaskValue,
    ) -> Result<Vec<NodeValueIndex>> {
        let mut node = self
            .nodes
            .get_mut(node_index)
            .ok_or_else(|| VegaFusionError::internal("Missing node"))?;
        if !matches!(node.task().task_kind(), TaskKind::Value(_)) {
            return Err(VegaFusionError::internal(
                "Task with index {} is not a Value",
            ));
        }

        node.task = Some(Task {
            variable: node.task().variable.clone(),
            scope: node.task().scope.clone(),
            task_kind: Some(TaskKind::Value(ProtoTaskValue::try_from(&value)?)),
        });

        let mut node_value_indexes = Vec::new();
        for node_index in self.update_state_fingerprints()? {
            node_value_indexes.push(NodeValueIndex::new(node_index as u32, None));

            for output_index in 0..self
                .nodes
                .get(node_index as usize)
                .unwrap()
                .task()
                .output_vars()
                .len()
            {
                node_value_indexes.push(NodeValueIndex::new(
                    node_index as u32,
                    Some(output_index as u32),
                ));
            }
        }
        Ok(node_value_indexes)
    }

    pub fn parent_nodes(&self, node_index: usize) -> Result<Vec<&TaskNode>> {
        let node = self
            .nodes
            .get(node_index)
            .with_context(|| format!("Node index {} out of bounds", node_index))?;
        Ok(node
            .incoming
            .iter()
            .map(|edge| self.nodes.get(edge.source as usize).unwrap())
            .collect())
    }

    pub fn parent_indices(&self, node_index: usize) -> Result<Vec<usize>> {
        let node = self
            .nodes
            .get(node_index)
            .with_context(|| format!("Node index {} out of bounds", node_index))?;
        Ok(node
            .incoming
            .iter()
            .map(|edge| edge.source as usize)
            .collect())
    }

    pub fn child_nodes(&self, node_index: usize) -> Result<Vec<&TaskNode>> {
        let node = self
            .nodes
            .get(node_index)
            .with_context(|| format!("Node index {} out of bounds", node_index))?;
        Ok(node
            .outgoing
            .iter()
            .map(|edge| self.nodes.get(edge.target as usize).unwrap())
            .collect())
    }

    pub fn child_indices(&self, node_index: usize) -> Result<Vec<usize>> {
        let node = self
            .nodes
            .get(node_index)
            .with_context(|| format!("Node index {} out of bounds", node_index))?;
        Ok(node
            .outgoing
            .iter()
            .map(|edge| edge.target as usize)
            .collect())
    }

    pub fn node(&self, node_index: usize) -> Result<&TaskNode> {
        self.nodes
            .get(node_index)
            .with_context(|| format!("Node index {} out of bounds", node_index))
    }
}

impl NodeValueIndex {
    pub fn new(node_index: u32, output_index: Option<u32>) -> Self {
        Self {
            node_index,
            output_index,
        }
    }
}

impl TaskNode {
    pub fn task(&self) -> &Task {
        self.task.as_ref().unwrap()
    }
}
