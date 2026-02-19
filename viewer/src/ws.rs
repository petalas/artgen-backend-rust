use std::cell::RefCell;

use leptos::prelude::*;
use wasm_bindgen::prelude::*;
use web_sys::{MessageEvent, WebSocket};

#[derive(Clone, Debug)]
pub struct ViewerState {
    pub reference_image_b64: String,
    pub generated_image_b64: String,
    pub fitness: f64,
    pub polygons: u32,
    pub improvements: u64,
    pub evals_per_sec: f64,
    pub total_evals: u64,
    pub elapsed_secs: u64,
    pub paused: bool,
    pub connected: bool,
    pub connecting: bool,
    pub drawing_json: Option<String>,
    pub image_width: u32,
    pub image_height: u32,
}

impl Default for ViewerState {
    fn default() -> Self {
        Self {
            reference_image_b64: String::new(),
            generated_image_b64: String::new(),
            fitness: 0.0,
            polygons: 0,
            improvements: 0,
            evals_per_sec: 0.0,
            total_evals: 0,
            elapsed_secs: 0,
            paused: false,
            connected: false,
            connecting: true,
            drawing_json: None,
            image_width: 0,
            image_height: 0,
        }
    }
}

thread_local! {
    static WS_HANDLE: RefCell<Option<WebSocket>> = const { RefCell::new(None) };
}

pub fn send_ws_command(cmd_type: &str) {
    WS_HANDLE.with(|ws| {
        if let Some(ws) = ws.borrow().as_ref() {
            let msg = serde_json::json!({ "type": cmd_type });
            ws.send_with_str(&msg.to_string()).ok();
        }
    });
}

pub fn connect_ws(state: RwSignal<ViewerState>) {
    let window = web_sys::window().expect("no window");
    let hostname = window.location().hostname().unwrap_or_else(|_| "localhost".into());
    let url = format!("ws://{}:9001", hostname);

    let ws = match WebSocket::new(&url) {
        Ok(ws) => ws,
        Err(_) => {
            schedule_reconnect(state);
            return;
        }
    };

    state.update(|s| {
        s.connecting = true;
        s.connected = false;
    });

    // onopen
    let state_open = state;
    let on_open = Closure::<dyn Fn()>::new(move || {
        state_open.update(|s| {
            s.connected = true;
            s.connecting = false;
        });
    });
    ws.set_onopen(Some(on_open.as_ref().unchecked_ref()));
    on_open.forget();

    // onmessage
    let state_msg = state;
    let on_message = Closure::<dyn Fn(MessageEvent)>::new(move |e: MessageEvent| {
        if let Some(text) = e.data().as_string() {
            if let Ok(data) = serde_json::from_str::<serde_json::Value>(&text) {
                handle_message(&data, state_msg);
            }
        }
    });
    ws.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
    on_message.forget();

    // onclose
    let state_close = state;
    let on_close = Closure::<dyn Fn()>::new(move || {
        state_close.update(|s| {
            s.connected = false;
            s.connecting = false;
        });
        WS_HANDLE.with(|h| h.borrow_mut().take());
        schedule_reconnect(state_close);
    });
    ws.set_onclose(Some(on_close.as_ref().unchecked_ref()));
    on_close.forget();

    // onerror — just let onclose handle reconnect
    let ws_err = ws.clone();
    let on_error = Closure::<dyn Fn()>::new(move || {
        ws_err.close().ok();
    });
    ws.set_onerror(Some(on_error.as_ref().unchecked_ref()));
    on_error.forget();

    // Store ws handle in thread-local
    WS_HANDLE.with(|h| *h.borrow_mut() = Some(ws));
}

fn schedule_reconnect(state: RwSignal<ViewerState>) {
    gloo_timers::callback::Timeout::new(1000, move || {
        connect_ws(state);
    })
    .forget();
}

fn handle_message(data: &serde_json::Value, state: RwSignal<ViewerState>) {
    let msg_type = data["type"].as_str().unwrap_or("");

    state.update(|s| {
        if let Some(paused) = data["paused"].as_bool() {
            s.paused = paused;
        }
        if let Some(fitness) = data["fitness"].as_f64() {
            s.fitness = fitness;
        }
        if let Some(polygons) = data["polygons"].as_u64() {
            s.polygons = polygons as u32;
        }
        if let Some(improvements) = data["improvements"].as_u64() {
            s.improvements = improvements;
        }
        if let Some(eps) = data["evalsPerSec"].as_f64() {
            s.evals_per_sec = eps;
        }
        if let Some(te) = data["totalEvals"].as_u64() {
            s.total_evals = te;
        }
        if let Some(elapsed) = data["elapsed"].as_u64() {
            s.elapsed_secs = elapsed;
        }
        if let Some(w) = data["imageWidth"].as_u64() {
            s.image_width = w as u32;
        }
        if let Some(h) = data["imageHeight"].as_u64() {
            s.image_height = h as u32;
        }

        match msg_type {
            "init" => {
                if let Some(ref_img) = data["referenceImage"].as_str() {
                    s.reference_image_b64 = ref_img.to_string();
                }
                if let Some(img) = data["image"].as_str() {
                    s.generated_image_b64 = img.to_string();
                }
                if let Some(dj) = data["drawingJson"].as_str() {
                    s.drawing_json = Some(dj.to_string());
                }
            }
            "update" => {
                if let Some(img) = data["image"].as_str() {
                    s.generated_image_b64 = img.to_string();
                }
                if let Some(dj) = data["drawingJson"].as_str() {
                    s.drawing_json = Some(dj.to_string());
                }
            }
            "stats" => {
                // stats-only update, no image data
            }
            _ => {}
        }
    });
}
