use leptos::prelude::*;
use wasm_bindgen::prelude::*;

use crate::models::Drawing;
use crate::ws::{send_ws_command, ViewerState};

fn download_blob(content: &str, filename: &str, mime_type: &str) {
    let window = web_sys::window().expect("no window");
    let document = window.document().expect("no document");

    let arr = js_sys::Array::new();
    arr.push(&JsValue::from_str(content));

    let opts = web_sys::BlobPropertyBag::new();
    opts.set_type(mime_type);

    let blob = web_sys::Blob::new_with_str_sequence_and_options(&arr, &opts).expect("blob");
    let url = web_sys::Url::create_object_url_with_blob(&blob).expect("url");

    let a: web_sys::HtmlAnchorElement = document
        .create_element("a")
        .expect("create a")
        .dyn_into()
        .expect("into anchor");

    a.set_href(&url);
    a.set_download(filename);
    a.click();

    web_sys::Url::revoke_object_url(&url).ok();
}

#[component]
pub fn Controls(state: RwSignal<ViewerState>) -> impl IntoView {
    let pause_label = move || {
        if state.get().paused {
            "RESUME"
        } else {
            "PAUSE"
        }
    };

    let pause_class = move || {
        if state.get().paused {
            "btn btn-success"
        } else {
            "btn btn-warning"
        }
    };

    let on_pause = move |_| {
        let cmd = if state.get().paused { "resume" } else { "pause" };
        send_ws_command(cmd);
    };

    let on_export_json = move |_| {
        let s = state.get();
        if let Some(json) = &s.drawing_json {
            download_blob(json, "drawing.best.json", "application/json");
        }
    };

    let on_export_svg = move |_| {
        let s = state.get();
        if let Some(json) = &s.drawing_json {
            if let Ok(drawing) = serde_json::from_str::<Drawing>(json) {
                let svg = drawing.to_svg(s.image_width, s.image_height);
                download_blob(&svg, "drawing.svg", "image/svg+xml");
            }
        }
    };

    let has_drawing = move || state.get().drawing_json.is_some();
    let is_connected = move || state.get().connected;

    view! {
        <div class="controls-section">
            <button
                class={pause_class}
                on:click={on_pause}
                disabled={move || !is_connected()}
            >
                {pause_label}
            </button>
            <button
                class="btn btn-primary"
                on:click={on_export_json}
                disabled={move || !has_drawing()}
            >
                "EXPORT JSON"
            </button>
            <button
                class="btn btn-primary"
                on:click={on_export_svg}
                disabled={move || !has_drawing()}
            >
                "EXPORT SVG"
            </button>
        </div>
    }
}
