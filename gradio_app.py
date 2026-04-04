import os

import gradio as gr
import websockets

from app.services.gradio_service import GradioAPIService

# Configuration from environment variables
WS_URL = os.getenv("WS_URL", "ws://localhost:8000/ws/logs")

# Load available models from API
models_info = GradioAPIService.get_models()
PROVIDERS = models_info.get("providers", [])


async def stream_logs():
    """Connects to the WebSocket and yields logs."""
    try:
        async with websockets.connect(WS_URL) as websocket:
            while True:
                log_message = await websocket.recv()
                # Handle both str and bytes responses
                if isinstance(log_message, (bytes, bytearray, memoryview)):
                    log_message = log_message.decode("utf-8")
                yield log_message + "\n"
    except Exception as e:
        yield f"Connection error: {e}\n"


def get_repo_info(repo_id: str, token: str):
    """Fetch repository configuration and splits from Hugging Face."""
    configs_info, splits_info = GradioAPIService.get_repo_info(repo_id, token)

    if not configs_info:
        return gr.update(choices=["default"]), gr.update(choices=["train"])

    configs = configs_info.get("configs", ["default"])
    default_splits = splits_info.get("default_splits", ["train", "test", "validation"])

    return gr.update(choices=configs, value=configs[0] if configs else None), gr.update(
        choices=default_splits, value=default_splits[0] if default_splits else None
    )


def update_splits(repo_id: str, config_name: str, token: str):
    """Update available splits based on selected configuration."""
    splits = GradioAPIService.update_splits(repo_id, config_name, token)
    return gr.update(choices=splits, value=splits[0] if splits else "train")


def generate_data(
    prompt: str,
    files,
    demo_file,
    provider: str,
    model,
    api_key,
    temperature: float,
    count: int,
    agentic: bool,
    use_rag: bool,
    mcp_servers,
    conserve_tokens: bool,
    rate_limit,
    hf_repo,
    hf_token,
    hf_private: bool,
    hf_append: bool,
    hf_config,
    hf_split,
):
    """Generate synthetic data via the API."""
    return GradioAPIService.generate_data(
        prompt=prompt,
        files=files,
        demo_file=demo_file,
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
        count=count,
        agentic=agentic,
        use_rag=use_rag,
        mcp_servers=mcp_servers,
        conserve_tokens=conserve_tokens,
        rate_limit=rate_limit,
        hf_repo=hf_repo,
        hf_token=hf_token,
        hf_private=hf_private,
        hf_append=hf_append,
        hf_config=hf_config,
        hf_split=hf_split,
    )


def modify_hf_dataset(
    repo_id: str,
    token,
    config,
    split,
    data_json,
    operation: str,
    gen_mode: bool,
    gen_instruction,
    gen_new_col,
    gen_rows,
    provider: str,
    api_key,
) -> str:
    """Modify an existing Hugging Face dataset by adding columns or rows."""
    return GradioAPIService.modify_hf_dataset(
        repo_id=repo_id,
        token=token,
        config=config,
        split=split,
        data_json=data_json,
        operation=operation,
        gen_mode=gen_mode,
        gen_instruction=gen_instruction,
        gen_new_col=gen_new_col,
        gen_rows=gen_rows,
        provider=provider,
        api_key=api_key,
    )


# --- Settings Helper Functions ---
def refresh_models_info():
    """Refreshes the models info and returns updated choices."""
    info = GradioAPIService.get_models()
    providers = info.get("providers", [])
    configured = info.get("configured", [])
    # Mark configured providers
    display_providers = []
    for p in providers:
        if p in configured:
             display_providers.append(f"{p} (Configured)")
        else:
             display_providers.append(p)
    
    return gr.update(choices=providers), f"Refreshed. Configured: {', '.join(configured)}"

def save_token_ui(provider, token):
    if not token:
        return "Error: Token cannot be empty."
    return GradioAPIService.set_token(provider, token)

def delete_token_ui(provider):
    return GradioAPIService.delete_token(provider)

# --- UI Layout ---

with gr.Blocks(title="SLM Data Generator") as demo:
    gr.Markdown("# 🚀 Synthetic Data Generation Service")
    gr.Markdown("Generate, Optimize, and Export datasets for SLM training.")

    with gr.Tabs():
        with gr.TabItem("Generate Data"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Configuration")
                    with gr.Row():
                        provider_dropdown = gr.Dropdown(
                            choices=PROVIDERS, value="groq", label="LLM Provider"
                        )
                        refresh_providers_btn = gr.Button("🔄", size="sm", scale=0)
                    
                    api_key_input = gr.Textbox(
                        label="Provider API Key",
                        type="password",
                        placeholder="Optional override (if not set in Settings/Env)",
                    )

                    model_input = gr.Textbox(
                        label="Model Name (Optional)", placeholder="e.g. llama3-8b-8192"
                    )
                    temp_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                    )
                    count_slider = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=10,
                        step=1,
                        label="Number of Pairs",
                    )

                    with gr.Accordion("Advanced & Agentic", open=False):
                        with gr.Row():
                            agentic_checkbox = gr.Checkbox(
                                label="Agentic Mode", value=False
                            )
                            rag_checkbox = gr.Checkbox(label="Use RAG", value=False)
                        mcp_input = gr.Textbox(
                            label="MCP Servers (JSON List)",
                            placeholder='["http://server1"]',
                            lines=1,
                        )

                    with gr.Accordion("Optimization", open=False):
                        conserve_chk = gr.Checkbox(label="Conserve Tokens", value=False)
                        rate_slider = gr.Slider(
                            minimum=0,
                            maximum=60,
                            value=0,
                            step=1,
                            label="Rate Limit (req/min)",
                        )

                    with gr.Accordion("Hugging Face Export", open=True):
                        hf_repo_input = gr.Textbox(
                            label="Repo ID", placeholder="username/dataset"
                        )
                        hf_token_input = gr.Textbox(
                            label="Write Token", type="password"
                        )
                        with gr.Row():
                            hf_fetch_btn = gr.Button("Load Configs")
                        with gr.Row():
                            hf_config_dropdown = gr.Dropdown(
                                label="Config Name", allow_custom_value=True
                            )
                            hf_split_dropdown = gr.Dropdown(
                                label="Split", value="train", allow_custom_value=True
                            )
                        with gr.Row():
                            hf_private_chk = gr.Checkbox(label="Private", value=False)
                            hf_append_chk = gr.Checkbox(label="Append", value=False)

                    hf_fetch_btn.click(
                        get_repo_info,
                        inputs=[hf_repo_input, hf_token_input],
                        outputs=[hf_config_dropdown, hf_split_dropdown],
                    )

                    # Update splits when config changes
                    hf_config_dropdown.change(
                        update_splits,
                        inputs=[hf_repo_input, hf_config_dropdown, hf_token_input],
                        outputs=[hf_split_dropdown],
                    )

                    gr.Markdown("### 📄 Source Content")
                    files_upload = gr.File(
                        label="Upload Context Files", file_count="multiple"
                    )
                    demo_upload = gr.File(
                        label="Upload Few-Shot Demo (CSV)", file_count="single"
                    )

                    generate_btn = gr.Button(
                        "✨ Generate & Export", variant="primary", size="lg"
                    )

                    # Refresh providers action
                    refresh_providers_btn.click(
                        refresh_models_info,
                        outputs=[provider_dropdown, gr.Textbox(visible=False)] # hidden status
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### 📝 Prompt")
                    prompt_input = gr.Textbox(
                        label="Instructions",
                        placeholder="Describe what kind of data you want to generate...",
                        lines=4,
                        value="Generate Q&A pairs about the provided documents.",
                    )

                    gr.Markdown("### 📊 Results")
                    status_output = gr.Textbox(label="Status", interactive=False)

                    with gr.Tabs():
                        with gr.TabItem("Data Table"):
                            dataframe_output = gr.Dataframe(label="Generated Data")
                        with gr.TabItem("JSON View"):
                            json_output = gr.Code(language="json", label="Raw JSON")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📡 Live Server Logs")
                    logs_output = gr.Textbox(
                        label="Server Logs",
                        lines=10,
                        max_lines=15,
                        interactive=False,
                        autoscroll=True,
                    )
                    start_logs_btn = gr.Button("Connect to Logs")
                    start_logs_btn.click(stream_logs, inputs=None, outputs=logs_output)

            generate_btn.click(
                fn=generate_data,
                inputs=[
                    prompt_input,
                    files_upload,
                    demo_upload,
                    provider_dropdown,
                    model_input,
                    api_key_input,
                    temp_slider,
                    count_slider,
                    agentic_checkbox,
                    rag_checkbox,
                    mcp_input,
                    conserve_chk,
                    rate_slider,
                    hf_repo_input,
                    hf_token_input,
                    hf_private_chk,
                    hf_append_chk,
                    hf_config_dropdown,
                    hf_split_dropdown,
                ],
                outputs=[dataframe_output, json_output, status_output],
            )

        with gr.TabItem("Modify Dataset"):
            gr.Markdown("### 🛠 Modify Existing Hugging Face Dataset")
            with gr.Row():
                mod_repo = gr.Textbox(label="Repo ID")
                mod_token = gr.Textbox(label="Write Token", type="password")

            mod_fetch_btn = gr.Button("Load Dataset Info")
            with gr.Row():
                mod_config = gr.Dropdown(label="Config", allow_custom_value=True, multiselect=True)
                mod_split = gr.Dropdown(
                    label="Split", value=["train"], allow_custom_value=True, multiselect=True
                )

            mod_fetch_btn.click(
                get_repo_info,
                inputs=[mod_repo, mod_token],
                outputs=[mod_config, mod_split],
            )

            # Update splits when config changes
            # Note: with multiselect, config will be a list. update_splits needs to handle this?
            # get_repo_info returns all splits. update_splits filters?
            # For simplicity, let's keep update_splits simpler or update it.
            # If we select multiple configs, showing splits for ALL of them is tricky (union?).
            # Let's keep it simple: If multiple configs, maybe just show default splits or union.
            
            # We need to update user functions too for multiselect.
            
            mod_config.change(
                update_splits,
                inputs=[mod_repo, mod_config, mod_token],
                outputs=[mod_split],
            )

            mod_op = gr.Radio(
                ["append_rows", "add_column"], label="Operation", value="append_rows"
            )

            with gr.Accordion("Generative Options", open=True):
                mod_gen_mode = gr.Checkbox(label="Use LLM Generation", value=False)
                with gr.Row():
                    mod_provider = gr.Dropdown(
                        choices=PROVIDERS, value="groq", label="Provider"
                    )
                    mod_api_key = gr.Textbox(label="API Key", type="password")
                mod_instruction = gr.Textbox(
                    label="Instruction (Prompt)",
                    placeholder="For add_column: describe the new column. For append_rows: describe what rows to generate.",
                    lines=2,
                )
                with gr.Row():
                    mod_new_col = gr.Textbox(
                        label="New Column Name (required for add_column)"
                    )
                    mod_rows = gr.Number(
                        label="Number of Rows (for append_rows)", value=10, minimum=1
                    )

            # Dynamic label for manual data based on operation
            mod_data_label = gr.Textbox(
                label="Manual Data (JSON) - Only when NOT using LLM Generation",
                placeholder='For add_column: [{"value1": "data1"}, ...]\nFor append_rows: [{"col1": "val1", "col2": "val2"}, ...]',
                lines=5,
            )

            mod_btn = gr.Button("Apply Modification")
            mod_status = gr.Textbox(label="Result", interactive=False)

            mod_btn.click(
                fn=modify_hf_dataset,
                inputs=[
                    mod_repo,
                    mod_token,
                    mod_config,
                    mod_split,
                    mod_data_label,
                    mod_op,
                    mod_gen_mode,
                    mod_instruction,
                    mod_new_col,
                    mod_rows,
                    mod_provider,
                    mod_api_key,
                ],
                outputs=[mod_status],
            )
            
        with gr.TabItem("Settings"):
            gr.Markdown("### 🔑 API Key Management")
            gr.Markdown("Configure API keys for different providers. These are stored locally in `tokens.db`.")
            
            with gr.Row():
                with gr.Column():
                    settings_provider = gr.Dropdown(
                        choices=["openai", "groq", "google", "huggingface", "huggingface-inference"], 
                        label="Provider",
                        value="openai"
                    )
                    settings_token = gr.Textbox(label="API Key / Token", type="password")
                    with gr.Row():
                        settings_save_btn = gr.Button("Save Token", variant="primary")
                        settings_del_btn = gr.Button("Delete Token", variant="stop")
                    
                    settings_status = gr.Textbox(label="Status", interactive=False)
            
            settings_save_btn.click(
                save_token_ui,
                inputs=[settings_provider, settings_token],
                outputs=[settings_status]
            )
            
            settings_del_btn.click(
                delete_token_ui,
                inputs=[settings_provider],
                outputs=[settings_status]
            )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
