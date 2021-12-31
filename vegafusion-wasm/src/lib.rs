mod utils;

use prost::Message;

// use vegafusion_core::expression::parser::parse;

use std::convert::TryFrom;
use vegafusion_core::data::scalar::{ScalarValue, ScalarValueHelpers};
use vegafusion_core::proto::gen::tasks::{
    NodeValueIndex, TaskGraph, TaskGraphValueRequest, TaskGraphValueResponse, VariableNamespace,
};
use vegafusion_core::task_graph::task_value::TaskValue;
use wasm_bindgen::prelude::*;

use vegafusion_core::error::Result;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use vegafusion_core::data::table::VegaFusionTable;
use vegafusion_core::planning::extract::extract_server_data;
use vegafusion_core::planning::stitch::{stitch_specs, CommPlan};
use vegafusion_core::proto::gen::expression::literal::Value;
use vegafusion_core::proto::gen::services::{
    vega_fusion_runtime_request, vega_fusion_runtime_response, VegaFusionRuntimeRequest,
    VegaFusionRuntimeResponse,
};
use vegafusion_core::spec::chart::ChartSpec;
use vegafusion_core::task_graph::task_graph::ScopedVariable;
use web_sys::Element;
use vegafusion_core::planning::optimize_server::split_data_url_nodes;
use vegafusion_core::planning::watch::WatchPlan;
use wasm_bindgen_futures::JsFuture;
use js_sys::Promise;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

pub fn set_panic_hook() {
    // When the `console_error_panic_hook` feature is enabled, we can call the
    // `set_panic_hook` function at least once during initialization, and then
    // we will get better error messages if our code ever panics.
    //
    // For more details see
    // https://github.com/rustwasm/console_error_panic_hook#readme
    #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();
}

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);

    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen]
pub struct MsgReceiver {
    spec: ChartSpec,
    server_spec: ChartSpec,
    comm_plan: CommPlan,
    send_msg_fn: Arc<js_sys::Function>,
    task_graph: Arc<Mutex<TaskGraph>>,
    task_graph_mapping: HashMap<ScopedVariable, NodeValueIndex>,
    server_to_client_value_indices: Arc<HashSet<NodeValueIndex>>,
    view: View,
    verbose: bool,
    debounce_wait: f64,
    debounce_max_wait: Option<f64>,
}

#[wasm_bindgen]
impl MsgReceiver {
    fn new(
        element: Element,
        spec: ChartSpec,
        server_spec: ChartSpec,
        comm_plan: CommPlan,
        task_graph: TaskGraph,
        send_msg_fn: js_sys::Function,
        verbose: bool,
        debounce_wait: f64,
        debounce_max_wait: Option<f64>,
    ) -> Self {
        set_panic_hook();

        let task_graph_mapping = task_graph.build_mapping();

        let server_to_client_value_indices: Arc<HashSet<_>> = Arc::new(
            comm_plan
                .server_to_client
                .iter()
                .map(|scoped_var| task_graph_mapping.get(scoped_var).unwrap().clone())
                .collect(),
        );

        // Mount vega chart
        let window = web_sys::window().expect("no global `window` exists");
        let document = window.document().expect("should have a document on window");
        let dataflow = parse(JsValue::from_serde(&spec).unwrap());

        let view = View::new(dataflow);
        view.initialize(element);
        view.hover();
        setup_tooltip(&view);

        let this = Self {
            spec,
            server_spec,
            comm_plan,
            task_graph: Arc::new(Mutex::new(task_graph)),
            task_graph_mapping,
            send_msg_fn: Arc::new(send_msg_fn),
            server_to_client_value_indices,
            view,
            verbose,
            debounce_wait,
            debounce_max_wait,
        };

        this.register_callbacks();

        this
    }

    pub fn receive(&mut self, bytes: Vec<u8>) {
        // Decode message
        let response = VegaFusionRuntimeResponse::decode(bytes.as_slice()).unwrap();

        if let Some(response) = response.response {
            match response {
                vega_fusion_runtime_response::Response::TaskGraphValues(task_graph_vals) => {
                    let view = self.view();
                    for response_val in task_graph_vals.response_values {
                        let value = response_val.value.expect("Unwrap value");
                        let scope = response_val.scope.as_slice();
                        let var = response_val.variable.expect("Unwrap variable");

                        // Convert from proto task value to task value
                        let value = TaskValue::try_from(&value).expect("Deserialize value");

                        match &value {
                            TaskValue::Scalar(value) => {
                                let json = value.to_json().unwrap();
                                if self.verbose {
                                    log(&format!("VegaFusion(wasm): Received {}", var.name));
                                    log(&serde_json::to_string_pretty(&json).unwrap());
                                }

                                let js_value =
                                    JsValue::from_serde(&json).unwrap();
                                set_signal_value(view, &var.name, scope, js_value);
                            }
                            TaskValue::Table(value) => {
                                let json = value.to_json();
                                if self.verbose {
                                    log(&format!("VegaFusion(wasm): Received {}", var.name));
                                    log(&serde_json::to_string_pretty(&json).unwrap());
                                }

                                let js_value = JsValue::from_serde(&value.to_json()).unwrap();
                                set_data_value(view, &var.name, scope, js_value);
                            }
                        }
                    }
                    view.run();
                }
                vega_fusion_runtime_response::Response::Error(error) => {
                    log(&error.msg());
                }
            }
        }
    }

    fn view(&self) -> &View {
        &self.view
    }

    fn add_signal_listener(&self, name: &str, scope: &[u32], handler: JsValue) {
        add_signal_listener(self.view(), name, scope, handler, self.debounce_wait, self.debounce_max_wait);
    }

    fn add_data_listener(&self, name: &str, scope: &[u32], handler: JsValue) {
        add_data_listener(self.view(), name, scope, handler, self.debounce_wait, self.debounce_max_wait);
    }

    fn register_callbacks(&self) {
        for scoped_var in &self.comm_plan.client_to_server {
            let var_name = scoped_var.0.name.clone();
            let scope = scoped_var.1.as_slice();
            let node_value_index = self.task_graph_mapping.get(scoped_var).unwrap().clone();
            let server_to_client = self.server_to_client_value_indices.clone();

            let task_graph = self.task_graph.clone();
            let send_msg_fn = self.send_msg_fn.clone();
            let verbose = self.verbose;

            // Register callbacks
            match scoped_var.0.namespace() {
                VariableNamespace::Signal => {
                    let closure = Closure::wrap(Box::new(move |name: String, val: JsValue| {
                        let val: serde_json::Value = val.into_serde().unwrap();
                        if verbose {
                            log(&format!("VegaFusion(wasm): Sending signal {}", name));
                            log(&serde_json::to_string_pretty(&val).unwrap());
                        }

                        let mut task_graph = task_graph.lock().unwrap();
                        let updated_nodes = &task_graph
                            .update_value(
                                node_value_index.node_index as usize,
                                TaskValue::Scalar(ScalarValue::from_json(&val).unwrap()),
                            )
                            .unwrap();

                        // Filter to update nodes in the comm plan
                        let updated_nodes: Vec<_> = updated_nodes
                            .iter()
                            .cloned()
                            .filter(|node| server_to_client.contains(node))
                            .collect();

                        let request_msg = VegaFusionRuntimeRequest {
                            request: Some(vega_fusion_runtime_request::Request::TaskGraphValues(
                                TaskGraphValueRequest {
                                    task_graph: Some(task_graph.clone()),
                                    indices: updated_nodes,
                                },
                            )),
                        };

                        Self::send_request(send_msg_fn.as_ref(), request_msg);
                    })
                        as Box<dyn FnMut(String, JsValue)>);

                    let ret_cb = closure.as_ref().clone();
                    closure.forget();

                    self.add_signal_listener(&var_name, scope, ret_cb);
                }
                VariableNamespace::Data => {
                    let closure = Closure::wrap(Box::new(move |name: String, val: JsValue| {
                        let val: serde_json::Value = val.into_serde().unwrap();
                        if verbose {
                            log(&format!("VegaFusion(wasm): Sending data {}", name));
                            log(&serde_json::to_string_pretty(&val).unwrap());
                        }

                        let mut task_graph = task_graph.lock().expect("lock task graph");

                        let updated_nodes = &task_graph
                            .update_value(
                                node_value_index.node_index as usize,
                                TaskValue::Table(VegaFusionTable::from_json(&val, 1024).unwrap()),
                            )
                            .unwrap();

                        // Filter to update nodes in the comm plan
                        let updated_nodes: Vec<_> = updated_nodes
                            .iter()
                            .cloned()
                            .filter(|node| server_to_client.contains(node))
                            .collect();

                        let request_msg = VegaFusionRuntimeRequest {
                            request: Some(vega_fusion_runtime_request::Request::TaskGraphValues(
                                TaskGraphValueRequest {
                                    task_graph: Some(task_graph.clone()),
                                    indices: updated_nodes,
                                },
                            )),
                        };

                        Self::send_request(send_msg_fn.as_ref(), request_msg);
                    })
                        as Box<dyn FnMut(String, JsValue)>);

                    let ret_cb = closure.as_ref().clone();
                    closure.forget();

                    self.add_data_listener(&var_name, scope, ret_cb);
                }
                _ => panic!("Unsupported namespace"),
            }
        }
    }

    fn send_request(send_msg_fn: &js_sys::Function, request_msg: VegaFusionRuntimeRequest) {
        let mut buf: Vec<u8> = Vec::new();
        buf.reserve(request_msg.encoded_len());
        request_msg.encode(&mut buf).unwrap();

        let context: JsValue = JsValue::from_serde(&serde_json::Value::Null).unwrap();
        let js_buffer = js_sys::Uint8Array::from(buf.as_slice());
        send_msg_fn.call1(&context, &js_buffer);
    }

    fn initial_node_value_indices(&self) -> Vec<NodeValueIndex> {
        self.comm_plan
            .server_to_client
            .iter()
            .map(|scoped_var| self.task_graph_mapping.get(scoped_var).unwrap().clone())
            .collect()
    }

    pub fn client_spec_json(&self) -> String {
        serde_json::to_string_pretty(&self.spec).unwrap()
    }

    pub fn server_spec_json(&self) -> String {
        serde_json::to_string_pretty(&self.server_spec).unwrap()
    }

    pub fn comm_plan_json(&self) -> String {
        serde_json::to_string_pretty(&WatchPlan::from(self.comm_plan.clone())).unwrap()
    }

    pub fn to_image_url(&self, img_type: &str, scale_factor: Option<f64>) -> Promise {
        self.view.to_image_url(img_type, scale_factor.unwrap_or(1.0))
    }
}

#[wasm_bindgen]
pub fn render_vegafusion(
    element: Element,
    spec_str: &str,
    verbose: bool,
    debounce_wait: f64,
    debounce_max_wait: Option<f64>,
    send_msg_fn: js_sys::Function,
) -> MsgReceiver {
    let mut spec: ChartSpec = serde_json::from_str(spec_str).unwrap();

    // Get full spec's scope
    let mut task_scope = spec.to_task_scope().unwrap();

    let mut server_spec = extract_server_data(&mut spec, &mut task_scope).expect("Failed to extract_server_data");
    let comm_plan = stitch_specs(&task_scope, &mut server_spec, &mut spec).expect("Failed to stitch_specs");

    split_data_url_nodes(&mut server_spec).expect("Failed to split_data_url_nodes");
    let task_scope = server_spec.to_task_scope().expect("Failed to create task scope for server spec");

    let tasks = server_spec.to_tasks().unwrap();
    let task_graph = TaskGraph::new(tasks, &task_scope).unwrap();

    // Create closure to update chart from received messages
    let receiver = MsgReceiver::new(
        element,
        spec,
        server_spec,
        comm_plan,
        task_graph.clone(),
        send_msg_fn,
        verbose,
        debounce_wait,
        debounce_max_wait,
    );

    // Request initial values
    let updated_node_indices: Vec<_> = receiver.initial_node_value_indices();

    let request_msg = VegaFusionRuntimeRequest {
        request: Some(vega_fusion_runtime_request::Request::TaskGraphValues(
            TaskGraphValueRequest {
                task_graph: Some(task_graph),
                indices: updated_node_indices,
            },
        )),
    };

    MsgReceiver::send_request(receiver.send_msg_fn.as_ref(), request_msg);

    receiver
}

#[wasm_bindgen(module = "/js/vega_utils.js")]
extern "C" {
    fn vega_version() -> String;

    #[wasm_bindgen(js_name = "getSignalValue")]
    fn get_signal_value(view: &View, name: &str, scope: &[u32]) -> JsValue;

    #[wasm_bindgen(js_name = "setSignalValue")]
    fn set_signal_value(view: &View, name: &str, scope: &[u32], value: JsValue);

    #[wasm_bindgen(js_name = "getDataValue")]
    fn get_data_value(view: &View, name: &str, scope: &[u32]) -> JsValue;

    #[wasm_bindgen(js_name = "setDataValue")]
    pub fn set_data_value(view: &View, name: &str, scope: &[u32], value: JsValue);

    #[wasm_bindgen(js_name = "addSignalListener")]
    fn add_signal_listener(view: &View, name: &str, scope: &[u32], handler: JsValue, wait: f64, maxWait: Option<f64>);

    #[wasm_bindgen(js_name = "addDataListener")]
    fn add_data_listener(view: &View, name: &str, scope: &[u32], handler: JsValue, wait: f64, maxWait: Option<f64>);

    #[wasm_bindgen(js_name = "setupTooltip")]
    fn setup_tooltip(view: &View);
}

#[wasm_bindgen(module = "vega")]
extern "C" {
    pub fn parse(spec: JsValue) -> JsValue;

    pub type View;

    #[wasm_bindgen(constructor)]
    pub fn new(dataflow: JsValue) -> View;

    #[wasm_bindgen(method, js_name = "initialize")]
    pub fn initialize(this: &View, container: Element);

    #[wasm_bindgen(method, js_name = "run")]
    pub fn run(this: &View);

    #[wasm_bindgen(method, js_name = "hover")]
    pub fn hover(this: &View);

    #[wasm_bindgen(method, js_name = "toImageURL")]
    pub fn to_image_url(this: &View, img_type: &str, scale_factor: f64) -> Promise;
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
