use leptos::prelude::*;
use wasm_bindgen::prelude::*;
use web_sys::HtmlInputElement;

use crate::ws::{send_ws_json, ViewerState};

#[component]
pub fn ProjectsPage(
    state: RwSignal<ViewerState>,
    page: RwSignal<String>,
) -> impl IntoView {
    let (new_name, set_new_name) = signal(String::new());
    let (creating, set_creating) = signal(false);

    let projects = move || state.get().projects;
    let active_project = move || state.get().active_project;
    let project_error = move || state.get().project_error;

    let on_switch = move |name: String| {
        send_ws_json(&serde_json::json!({
            "type": "switch_project",
            "name": name,
        }));
        page.set("evolution".into());
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

    let (renaming, set_renaming) = signal(Option::<String>::None);
    let (rename_value, set_rename_value) = signal(String::new());

    let on_rename_start = move |name: String| {
        set_rename_value.set(name.clone());
        set_renaming.set(Some(name));
    };

    let on_rename_cancel = move |_| {
        set_renaming.set(None);
    };

    let on_rename_submit = move |old_name: String| {
        let new_name = rename_value.get();
        if !new_name.is_empty() && new_name != old_name {
            send_ws_json(&serde_json::json!({
                "type": "rename_project",
                "oldName": old_name,
                "newName": new_name,
            }));
        }
        set_renaming.set(None);
    };

    let on_rename_input = move |ev: web_sys::Event| {
        let target: HtmlInputElement = ev.target().unwrap().dyn_into().unwrap();
        set_rename_value.set(target.value());
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
        });

        let onerror = Closure::<dyn Fn()>::new(move || {
            state.update(|s| s.project_error = Some("Failed to read file".into()));
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

        let onerror = Closure::<dyn Fn()>::new(move || {
            state.update(|s| s.project_error = Some("Failed to read JSON file".into()));
        });

        reader.set_onload(Some(onload.as_ref().unchecked_ref()));
        reader.set_onerror(Some(onerror.as_ref().unchecked_ref()));
        onload.forget();
        onerror.forget();

        reader.read_as_text(&file).ok();
    };

    use base64::Engine as _;

    view! {
        <div class="projects-page">
            <div class="projects-header">
                <h2>"Projects"</h2>
            </div>

            // Error display
            {move || project_error().map(|err| view! {
                <div class="project-error">{err}</div>
            })}

            // New project form
            <div class="new-project-form">
                <input
                    type="text"
                    class="project-name-input"
                    placeholder="New project name"
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

            // Project list
            <div class="projects-list">
                {move || {
                    let active = active_project();
                    let list = projects();
                    if list.is_empty() {
                        vec![view! {
                            <div class="projects-empty">
                                "No projects yet. Create one above."
                            </div>
                        }.into_any()]
                    } else {
                        let current_renaming = renaming.get();
                        list.into_iter().map(|p| {
                            let name = p.name.clone();
                            let is_active = active.as_deref() == Some(&name);
                            let is_renaming = current_renaming.as_deref() == Some(&name);
                            let name_switch = name.clone();
                            let name_delete = name.clone();
                            let name_reset = name.clone();
                            let name_rename = name.clone();
                            let name_rename_submit = name.clone();
                            let fitness_str = p.fitness
                                .map(|f| format!("{:.2}%", f))
                                .unwrap_or_else(|| "new".to_string());

                            if is_renaming {
                                view! {
                                    <div class={if is_active { "project-card active" } else { "project-card" }}>
                                        <div class="project-card-rename">
                                            <input
                                                type="text"
                                                class="project-rename-input"
                                                maxlength="64"
                                                prop:value={move || rename_value.get()}
                                                on:input={on_rename_input}
                                                on:keydown={move |ev: web_sys::KeyboardEvent| {
                                                    if ev.key() == "Enter" {
                                                        on_rename_submit(name_rename_submit.clone());
                                                    } else if ev.key() == "Escape" {
                                                        set_renaming.set(None);
                                                    }
                                                }}
                                            />
                                            <button
                                                class="btn-icon"
                                                title="Confirm rename"
                                                on:click={let n = name.clone(); move |_| on_rename_submit(n.clone())}
                                            >
                                                "\u{2713}"
                                            </button>
                                            <button
                                                class="btn-icon"
                                                title="Cancel"
                                                on:click={on_rename_cancel}
                                            >
                                                "\u{2715}"
                                            </button>
                                        </div>
                                    </div>
                                }.into_any()
                            } else {
                                view! {
                                    <div class={if is_active { "project-card active" } else { "project-card" }}>
                                        <div class="project-card-info"
                                             on:click={move |_| on_switch(name_switch.clone())}>
                                            <span class="project-card-name">{name.clone()}</span>
                                            <span class="project-card-fitness">{fitness_str}</span>
                                            {if is_active {
                                                Some(view! { <span class="project-card-badge">"active"</span> })
                                            } else {
                                                None
                                            }}
                                        </div>
                                        <div class="project-card-actions">
                                            <button
                                                class="btn-icon"
                                                title="Rename project"
                                                on:click={move |_| on_rename_start(name_rename.clone())}
                                            >
                                                "\u{270E}"
                                            </button>
                                            <button
                                                class="btn-icon"
                                                title="Reset progress"
                                                on:click={move |_| on_reset(name_reset.clone())}
                                            >
                                                "\u{21BB}"
                                            </button>
                                            <button
                                                class="btn-icon btn-icon-danger"
                                                title="Delete project"
                                                disabled={is_active}
                                                on:click={move |_| on_delete(name_delete.clone())}
                                            >
                                                "\u{2715}"
                                            </button>
                                        </div>
                                    </div>
                                }.into_any()
                            }
                        }).collect::<Vec<_>>()
                    }
                }}
            </div>

            // Import JSON (only if active project exists)
            {move || active_project().map(|name| view! {
                <div class="projects-import-section">
                    <label class="project-import-label">
                        {format!("Import .best.json into \"{}\"", name)}
                        <input
                            type="file"
                            accept=".json"
                            style="display:none"
                            on:change={on_import_file}
                        />
                    </label>
                </div>
            })}
        </div>
    }
}
