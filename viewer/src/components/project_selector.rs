use leptos::prelude::*;
use wasm_bindgen::prelude::*;
use web_sys::HtmlInputElement;

use crate::ws::{send_ws_json, ViewerState};

#[component]
pub fn ProjectSelector(state: RwSignal<ViewerState>) -> impl IntoView {
    let (panel_open, set_panel_open) = signal(false);
    let (new_name, set_new_name) = signal(String::new());
    let (creating, set_creating) = signal(false);

    let active_label = move || {
        let s = state.get();
        s.active_project
            .unwrap_or_else(|| "No project".to_string())
    };

    let toggle_panel = move |_| {
        set_panel_open.update(|v| *v = !*v);
        // Clear error when opening panel
        state.update(|s| s.project_error = None);
    };

    let projects = move || state.get().projects;
    let active_project = move || state.get().active_project;
    let project_error = move || state.get().project_error;

    let on_switch = move |name: String| {
        send_ws_json(&serde_json::json!({
            "type": "switch_project",
            "name": name,
        }));
        set_panel_open.set(false);
    };

    let on_delete = move |name: String| {
        send_ws_json(&serde_json::json!({
            "type": "delete_project",
            "name": name,
        }));
    };

    let on_reset = move |name: String| {
        send_ws_json(&serde_json::json!({
            "type": "reset_project",
            "name": name,
        }));
    };

    let on_name_input = move |ev: web_sys::Event| {
        let target: HtmlInputElement = ev.target().unwrap().dyn_into().unwrap();
        set_new_name.set(target.value());
    };

    let on_file_selected = move |ev: web_sys::Event| {
        let target: HtmlInputElement = ev.target().unwrap().dyn_into().unwrap();
        let Some(files) = target.files() else { return };
        let Some(file) = files.get(0) else { return };

        let name = new_name.get();
        if name.is_empty() {
            state.update(|s| s.project_error = Some("Please enter a project name".into()));
            return;
        }

        set_creating.set(true);
        state.update(|s| s.project_error = None);

        let reader = web_sys::FileReader::new().unwrap();
        let reader_clone = reader.clone();
        let state_clone = state;

        let onload = Closure::<dyn Fn()>::new(move || {
            let result = reader_clone.result().unwrap();
            let array_buffer = result.dyn_into::<js_sys::ArrayBuffer>().unwrap();
            let uint8_array = js_sys::Uint8Array::new(&array_buffer);
            let bytes = uint8_array.to_vec();

            let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);

            send_ws_json(&serde_json::json!({
                "type": "create_project",
                "name": name,
                "referenceImage": b64,
            }));

            set_creating.set(false);
            set_new_name.set(String::new());

            // Clear the file input
            // (The input is not easily accessible here, but new_name is cleared)
        });

        let state_err = state_clone;
        let onerror = Closure::<dyn Fn()>::new(move || {
            state_err.update(|s| s.project_error = Some("Failed to read file".into()));
            set_creating.set(false);
        });

        reader.set_onload(Some(onload.as_ref().unchecked_ref()));
        reader.set_onerror(Some(onerror.as_ref().unchecked_ref()));
        onload.forget();
        onerror.forget();

        reader.read_as_array_buffer(&file).ok();
    };

    let on_import_file = move |ev: web_sys::Event| {
        let target: HtmlInputElement = ev.target().unwrap().dyn_into().unwrap();
        let Some(files) = target.files() else { return };
        let Some(file) = files.get(0) else { return };

        let active = active_project();
        let Some(project_name) = active else {
            state.update(|s| s.project_error = Some("No active project to import into".into()));
            return;
        };

        let reader = web_sys::FileReader::new().unwrap();
        let reader_clone = reader.clone();

        let onload = Closure::<dyn Fn()>::new(move || {
            let result = reader_clone.result().unwrap();
            let text = result.as_string().unwrap_or_default();

            send_ws_json(&serde_json::json!({
                "type": "import_drawing",
                "name": project_name,
                "drawingJson": text,
            }));
        });

        let state_err = state;
        let onerror = Closure::<dyn Fn()>::new(move || {
            state_err.update(|s| s.project_error = Some("Failed to read JSON file".into()));
        });

        reader.set_onload(Some(onload.as_ref().unchecked_ref()));
        reader.set_onerror(Some(onerror.as_ref().unchecked_ref()));
        onload.forget();
        onerror.forget();

        reader.read_as_text(&file).ok();
    };

    use base64::Engine as _;

    view! {
        <div class="project-selector">
            <button class="project-toggle" on:click={toggle_panel}>
                {active_label}
                <span class="project-toggle-arrow">{move || if panel_open.get() { "\u{25B2}" } else { "\u{25BC}" }}</span>
            </button>

            {move || panel_open.get().then(|| view! {
                <div class="project-panel">
                    // Error display
                    {move || project_error().map(|err| view! {
                        <div class="project-error">{err}</div>
                    })}

                    // Project list
                    <div class="project-list">
                        {move || {
                            let active = active_project();
                            projects().into_iter().map(|p| {
                                let name = p.name.clone();
                                let is_active = active.as_deref() == Some(&name);
                                let name_switch = name.clone();
                                let name_delete = name.clone();
                                let name_reset = name.clone();
                                let fitness_str = p.fitness.map(|f| format!("{:.2}%", f)).unwrap_or_else(|| "-".to_string());

                                view! {
                                    <div class={if is_active { "project-item active" } else { "project-item" }}>
                                        <div class="project-item-info"
                                             on:click={move |_| if !is_active { on_switch(name_switch.clone()) }}>
                                            <span class="project-item-name">{name.clone()}</span>
                                            <span class="project-item-fitness">{fitness_str}</span>
                                        </div>
                                        <div class="project-item-actions">
                                            <button
                                                class="project-btn project-btn-reset"
                                                title="Reset"
                                                on:click={move |_| on_reset(name_reset.clone())}
                                            >
                                                "\u{21BB}"
                                            </button>
                                            <button
                                                class="project-btn project-btn-delete"
                                                title="Delete"
                                                disabled={is_active}
                                                on:click={move |_| on_delete(name_delete.clone())}
                                            >
                                                "\u{2715}"
                                            </button>
                                        </div>
                                    </div>
                                }
                            }).collect::<Vec<_>>()
                        }}
                    </div>

                    // Import JSON
                    {move || active_project().map(|_| view! {
                        <div class="project-import">
                            <label class="project-import-label">
                                "Import .best.json"
                                <input
                                    type="file"
                                    accept=".json"
                                    style="display:none"
                                    on:change={on_import_file}
                                />
                            </label>
                        </div>
                    })}

                    // New project form
                    <div class="new-project-form">
                        <input
                            type="text"
                            class="project-name-input"
                            placeholder="Project name"
                            maxlength="64"
                            prop:value={move || new_name.get()}
                            on:input={on_name_input}
                        />
                        <label class="project-file-label" class:disabled={move || creating.get() || new_name.get().is_empty()}>
                            {move || if creating.get() { "Creating..." } else { "Upload Image" }}
                            <input
                                type="file"
                                accept="image/*"
                                style="display:none"
                                disabled={move || creating.get() || new_name.get().is_empty()}
                                on:change={on_file_selected}
                            />
                        </label>
                    </div>
                </div>
            })}
        </div>
    }
}
