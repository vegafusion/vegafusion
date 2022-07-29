/*
 * VegaFusion
 * Copyright (C) 2022 VegaFusion Technologies LLC
 *
 * This program is distributed under multiple licenses.
 * Please consult the license documentation provided alongside
 * this program the details of the active license.
 */
use prost::Message;

// use vegafusion_core::expression::parser::parse;

use vegafusion_core::data::scalar::{ScalarValue, ScalarValueHelpers};
use vegafusion_core::proto::gen::tasks::{
    NodeValueIndex, TaskGraph, TaskGraphValueRequest, TzConfig, VariableNamespace,
};
use vegafusion_core::task_graph::task_value::TaskValue;
use wasm_bindgen::prelude::*;

use js_sys::Promise;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use vegafusion_core::data::table::VegaFusionTable;

use vegafusion_core::planning::stitch::CommPlan;
use vegafusion_core::planning::watch::WatchPlan;

use vegafusion_core::proto::gen::services::{
    query_request, query_result, QueryRequest, QueryResult,
};
use vegafusion_core::spec::chart::ChartSpec;
use vegafusion_core::task_graph::graph::ScopedVariable;

use vegafusion_core::planning::plan::SpecPlan;
use web_sys::Element;

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
#[derive(Clone)]
pub struct MsgReceiver {
    spec: Arc<ChartSpec>,
    server_spec: Arc<ChartSpec>,
    comm_plan: CommPlan,
    send_msg_fn: Arc<js_sys::Function>,
    task_graph: Arc<Mutex<TaskGraph>>,
    task_graph_mapping: Arc<HashMap<ScopedVariable, NodeValueIndex>>,
    server_to_client_value_indices: Arc<HashSet<NodeValueIndex>>,
    view: Arc<View>,
    verbose: bool,
    debounce_wait: f64,
    debounce_max_wait: Option<f64>,
}

#[wasm_bindgen]
impl MsgReceiver {
    #[allow(clippy::too_many_arguments)]
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
        let _document = window.document().expect("should have a document on window");
        let dataflow = parse(
            js_sys::JSON::parse(
                &serde_json::to_string(&spec).expect("Failed to parse spec as JSON"),
            )
            .unwrap(),
        );

        let view = View::new(dataflow);
        view.initialize(element);
        view.hover();
        setup_tooltip(&view);

        let this = Self {
            spec: Arc::new(spec),
            server_spec: Arc::new(server_spec),
            comm_plan,
            task_graph: Arc::new(Mutex::new(task_graph)),
            task_graph_mapping: Arc::new(task_graph_mapping),
            send_msg_fn: Arc::new(send_msg_fn),
            server_to_client_value_indices,
            view: Arc::new(view),
            verbose,
            debounce_wait,
            debounce_max_wait,
        };

        this.register_callbacks();

        this
    }

    pub fn receive(&mut self, bytes: Vec<u8>) {
        // Decode message
        let response = QueryResult::decode(bytes.as_slice()).unwrap();

        if let Some(response) = response.response {
            match response {
                query_result::Response::TaskGraphValues(task_graph_vals) => {
                    let view = self.view();
                    for (var, scope, value) in task_graph_vals
                        .deserialize()
                        .expect("Failed to deserialize response")
                    {
                        match &value {
                            TaskValue::Scalar(value) => {
                                let json = value.to_json().unwrap();
                                if self.verbose {
                                    log(&format!("VegaFusion(wasm): Received {}", var.name));
                                    log(&serde_json::to_string_pretty(&json).unwrap());
                                    log(&format!("DataType: {:#?}", &value.get_datatype()));
                                }

                                let js_value =
                                    js_sys::JSON::parse(&serde_json::to_string(&json).unwrap())
                                        .unwrap();
                                set_signal_value(view, &var.name, scope.as_slice(), js_value);
                            }
                            TaskValue::Table(value) => {
                                let json = value.to_json();
                                if self.verbose {
                                    log(&format!("VegaFusion(wasm): Received {}", var.name));
                                    log(&serde_json::to_string_pretty(&json).unwrap());
                                    log(&format!("Schema: {:#?}", &value.schema));
                                }

                                let js_value = js_sys::JSON::parse(
                                    &serde_json::to_string(&value.to_json()).unwrap(),
                                )
                                .unwrap();

                                set_data_value(view, &var.name, scope.as_slice(), js_value);
                            }
                        }
                    }
                    view.run();
                }
                query_result::Response::Error(error) => {
                    log(&error.msg());
                }
            }
        }
    }

    fn view(&self) -> &View {
        &self.view
    }

    fn add_signal_listener(&self, name: &str, scope: &[u32], handler: JsValue) {
        add_signal_listener(
            self.view(),
            name,
            scope,
            handler,
            self.debounce_wait,
            self.debounce_max_wait,
        );
    }

    fn add_data_listener(&self, name: &str, scope: &[u32], handler: JsValue) {
        add_data_listener(
            self.view(),
            name,
            scope,
            handler,
            self.debounce_wait,
            self.debounce_max_wait,
        );
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
            let this = self.clone();
            match scoped_var.0.namespace() {
                VariableNamespace::Signal => {
                    let closure = Closure::wrap(Box::new(move |name: String, val: JsValue| {
                        let val: serde_json::Value = serde_json::from_str(
                            &js_sys::JSON::stringify(&val).unwrap().as_string().unwrap(),
                        )
                        .unwrap();
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

                        let request_msg = QueryRequest {
                            request: Some(query_request::Request::TaskGraphValues(
                                TaskGraphValueRequest {
                                    task_graph: Some(task_graph.clone()),
                                    indices: updated_nodes,
                                },
                            )),
                        };

                        this.send_request(send_msg_fn.as_ref(), request_msg);
                    })
                        as Box<dyn FnMut(String, JsValue)>);

                    let ret_cb = closure.as_ref().clone();
                    closure.forget();

                    self.add_signal_listener(&var_name, scope, ret_cb);
                }
                VariableNamespace::Data => {
                    let closure = Closure::wrap(Box::new(move |name: String, val: JsValue| {
                        let val: serde_json::Value = serde_json::from_str(
                            &js_sys::JSON::stringify(&val).unwrap().as_string().unwrap(),
                        )
                        .unwrap();
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

                        if !updated_nodes.is_empty() {
                            let request_msg = QueryRequest {
                                request: Some(query_request::Request::TaskGraphValues(
                                    TaskGraphValueRequest {
                                        task_graph: Some(task_graph.clone()),
                                        indices: updated_nodes,
                                    },
                                )),
                            };

                            this.send_request(send_msg_fn.as_ref(), request_msg);
                        }
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

    fn send_request(&self, send_msg_fn: &js_sys::Function, request_msg: QueryRequest) {
        let mut buf: Vec<u8> = Vec::new();
        buf.reserve(request_msg.encoded_len());
        request_msg.encode(&mut buf).unwrap();

        let context =
            js_sys::JSON::parse(&serde_json::to_string(&serde_json::Value::Null).unwrap()).unwrap();

        let js_buffer = js_sys::Uint8Array::from(buf.as_slice());
        send_msg_fn
            .call2(&context, &js_buffer, &self.clone().into())
            .expect("send_request function call failed");
    }

    fn initial_node_value_indices(&self) -> Vec<NodeValueIndex> {
        self.comm_plan
            .server_to_client
            .iter()
            .map(|scoped_var| self.task_graph_mapping.get(scoped_var).unwrap().clone())
            .collect()
    }

    pub fn client_spec_json(&self) -> String {
        serde_json::to_string_pretty(self.spec.as_ref()).unwrap()
    }

    pub fn server_spec_json(&self) -> String {
        serde_json::to_string_pretty(self.server_spec.as_ref()).unwrap()
    }

    pub fn comm_plan_json(&self) -> String {
        serde_json::to_string_pretty(&WatchPlan::from(self.comm_plan.clone())).unwrap()
    }

    pub fn to_image_url(&self, img_type: &str, scale_factor: Option<f64>) -> Promise {
        self.view
            .to_image_url(img_type, scale_factor.unwrap_or(1.0))
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
    let spec: ChartSpec = serde_json::from_str(spec_str).unwrap();
    let spec_plan = SpecPlan::try_new(&spec, &Default::default()).unwrap();

    let task_scope = spec_plan
        .server_spec
        .to_task_scope()
        .expect("Failed to create task scope for server spec");

    let local_tz = local_timezone();
    let tz_config = TzConfig {
        local_tz,
        default_input_tz: None,
    };
    let tasks = spec_plan
        .server_spec
        .to_tasks(&tz_config, &Default::default())
        .unwrap();
    let task_graph = TaskGraph::new(tasks, &task_scope).unwrap();

    // Create closure to update chart from received messages
    let receiver = MsgReceiver::new(
        element,
        spec_plan.client_spec,
        spec_plan.server_spec,
        spec_plan.comm_plan,
        task_graph.clone(),
        send_msg_fn,
        verbose,
        debounce_wait,
        debounce_max_wait,
    );

    // Request initial values
    let updated_node_indices: Vec<_> = receiver.initial_node_value_indices();

    let request_msg = QueryRequest {
        request: Some(query_request::Request::TaskGraphValues(
            TaskGraphValueRequest {
                task_graph: Some(task_graph),
                indices: updated_node_indices,
            },
        )),
    };

    receiver.send_request(receiver.send_msg_fn.as_ref(), request_msg);

    receiver
}

#[wasm_bindgen]
pub fn vega_version() -> String {
    inner_vega_version()
}

#[wasm_bindgen]
pub fn make_grpc_send_message_fn(client: JsValue, hostname: String) -> js_sys::Function {
    inner_make_grpc_send_message_fn(client, hostname)
}

#[wasm_bindgen(module = "/js/vega_utils.js")]
extern "C" {
    #[wasm_bindgen(js_name = "vega_version")]
    fn inner_vega_version() -> String;

    #[wasm_bindgen(js_name = "localTimezone")]
    fn local_timezone() -> String;

    #[wasm_bindgen(js_name = "make_grpc_send_message_fn")]
    fn inner_make_grpc_send_message_fn(client: JsValue, hostname: String) -> js_sys::Function;

    #[wasm_bindgen(js_name = "getSignalValue")]
    fn get_signal_value(view: &View, name: &str, scope: &[u32]) -> JsValue;

    #[wasm_bindgen(js_name = "setSignalValue")]
    fn set_signal_value(view: &View, name: &str, scope: &[u32], value: JsValue);

    #[wasm_bindgen(js_name = "getDataValue")]
    fn get_data_value(view: &View, name: &str, scope: &[u32]) -> JsValue;

    #[wasm_bindgen(js_name = "setDataValue")]
    pub fn set_data_value(view: &View, name: &str, scope: &[u32], value: JsValue);

    #[wasm_bindgen(js_name = "addSignalListener")]
    fn add_signal_listener(
        view: &View,
        name: &str,
        scope: &[u32],
        handler: JsValue,
        wait: f64,
        maxWait: Option<f64>,
    );

    #[wasm_bindgen(js_name = "addDataListener")]
    fn add_data_listener(
        view: &View,
        name: &str,
        scope: &[u32],
        handler: JsValue,
        wait: f64,
        maxWait: Option<f64>,
    );

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
        println!("it works");
    }
}
