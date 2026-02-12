import gradio as gr
import requests
import json
import asyncio
import websockets
import pandas as pd
from typing import List

API_URL = "http://localhost:8000/api/v1"
WS_URL = "ws://localhost:8000/ws/logs"

def get_models():
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {"providers": ["groq", "openai", "google", "huggingface"], "defaults": {}}

models_info = get_models()
PROVIDERS = models_info.get("providers", [])

async def stream_logs():
    """Connects to the WebSocket and yields logs."""
    try:
        async with websockets.connect(WS_URL) as websocket:
            while True:
                log_message = await websocket.recv()
                yield log_message + "\n"
    except Exception as e:
        yield f"Connection error: {e}\n"

def get_repo_info(repo_id, token):
    if not repo_id:
        return gr.update(choices=[]), gr.update(choices=[])
    try:
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = requests.get(f"{API_URL}/dataset/info", params={"repo_id": repo_id, "token": token})
        if response.status_code == 200:
            info = response.json()
            configs = info.get("configs", ["default"])
            splits_map = info.get("splits", {})

            # If only default config, update splits
            splits = splits_map.get("default", ["train", "test", "validation"])
            return gr.update(choices=configs, value=configs[0] if configs else None), gr.update(choices=splits, value=splits[0] if splits else None)
        else:
            return gr.update(choices=["default"]), gr.update(choices=["train"])
    except:
        return gr.update(choices=["default"]), gr.update(choices=["train"])

def update_splits(repo_id, config_name, token):
    # This might require another call or logic if splits depend on config
    # We can try to fetch info again or cache it.
    # Simplest is just re-fetch info with config logic if needed, but get_dataset_info returns all.
    # But get_dataset_info logic I wrote returns all if possible.
    # Let's assume fetch_info handles it.
    pass

def generate_data(
    prompt,
    files,
    demo_file,
    provider,
    model,
    api_key,
    temperature,
    count,
    agentic,
    use_rag,
    mcp_servers,
    # Optimization
    conserve_tokens,
    rate_limit,
    # HF
    hf_repo,
    hf_token,
    hf_private,
    hf_append,
    hf_config,
    hf_split
):
    url = f"{API_URL}/generate"

    # Prepare form data
    data = {
        "prompt": prompt,
        "provider": provider,
        "temperature": float(temperature),
        "count": int(count),
        "agentic": agentic,
        "use_rag": use_rag,
        "mcp_servers": mcp_servers if mcp_servers else None,
        "conserve_tokens": conserve_tokens,
        "rate_limit": int(rate_limit) if rate_limit else 0,
        "hf_repo_id": hf_repo if hf_repo else None,
        "hf_token": hf_token if hf_token else None,
        "hf_private": hf_private,
        "hf_append": hf_append,
        "hf_config": hf_config,
        "hf_split": hf_split
    }

    if model:
        data["model"] = model
    if api_key:
        data["api_key"] = api_key

    files_payload = []
    if files:
        for f in files:
            files_payload.append(("files", (f.split("/")[-1], open(f, "rb"), "application/octet-stream")))

    if demo_file:
         files_payload.append(("demo_file", (demo_file.split("/")[-1], open(demo_file, "rb"), "text/csv")))

    try:
        response = requests.post(url, data=data, files=files_payload)

        if response.status_code == 200:
            result = response.json()
            df = pd.DataFrame(result.get("data", []))
            json_output = json.dumps(result.get("data", []), indent=2)
            msg = result.get("message", "Success")
            return df, json_output, msg
        else:
            try:
                detail = response.json().get("detail", response.text)
            except:
                detail = response.text
            return pd.DataFrame(), "{}", f"Error {response.status_code}: {detail}"

    except Exception as e:
         return pd.DataFrame(), "{}", f"Unexpected Error: {e}"

def modify_hf_dataset(repo_id, token, config, split, data_json, operation, gen_mode, gen_instruction, gen_new_col, gen_rows, provider, api_key):
    url = f"{API_URL}/dataset/modify"
    try:
        payload = {
            "repo_id": repo_id,
            "token": token,
            "operation": operation,
            "config_name": config,
            "split": split,
            "generative_mode": gen_mode,
            "instruction": gen_instruction,
            "new_column_name": gen_new_col,
            "num_rows": int(gen_rows) if gen_rows else 10,
            "provider": provider,
            "api_key": api_key
        }

        if not gen_mode:
             payload["data"] = json.loads(data_json)

        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return f"Success: {response.json().get('message')}"
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Failed: {e}"

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
                        provider_dropdown = gr.Dropdown(choices=PROVIDERS, value="groq", label="LLM Provider")
                        api_key_input = gr.Textbox(label="Provider API Key", type="password", placeholder="Overrides env var")

                    model_input = gr.Textbox(label="Model Name (Optional)", placeholder="e.g. llama3-8b-8192")
                    temp_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, label="Temperature")
                    count_slider = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="Number of Pairs")

                    with gr.Accordion("Advanced & Agentic", open=False):
                        with gr.Row():
                            agentic_checkbox = gr.Checkbox(label="Agentic Mode", value=False)
                            rag_checkbox = gr.Checkbox(label="Use RAG", value=False)
                        mcp_input = gr.Textbox(label="MCP Servers (JSON List)", placeholder='["http://server1"]', lines=1)

                    with gr.Accordion("Optimization", open=False):
                        conserve_chk = gr.Checkbox(label="Conserve Tokens", value=False)
                        rate_slider = gr.Slider(minimum=0, maximum=60, value=0, step=1, label="Rate Limit (req/min)")

                    with gr.Accordion("Hugging Face Export", open=True):
                        hf_repo_input = gr.Textbox(label="Repo ID", placeholder="username/dataset")
                        hf_token_input = gr.Textbox(label="Write Token", type="password")
                        with gr.Row():
                            hf_fetch_btn = gr.Button("Load Configs")
                        with gr.Row():
                            hf_config_dropdown = gr.Dropdown(label="Config Name", allow_custom_value=True)
                            hf_split_dropdown = gr.Dropdown(label="Split", value="train", allow_custom_value=True)
                        with gr.Row():
                            hf_private_chk = gr.Checkbox(label="Private", value=False)
                            hf_append_chk = gr.Checkbox(label="Append", value=False)

                    hf_fetch_btn.click(get_repo_info, inputs=[hf_repo_input, hf_token_input], outputs=[hf_config_dropdown, hf_split_dropdown])

                    gr.Markdown("### 📄 Source Content")
                    files_upload = gr.File(label="Upload Context Files", file_count="multiple")
                    demo_upload = gr.File(label="Upload Few-Shot Demo (CSV)", file_count="single")

                    generate_btn = gr.Button("✨ Generate & Export", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.Markdown("### 📝 Prompt")
                    prompt_input = gr.Textbox(
                        label="Instructions",
                        placeholder="Describe what kind of data you want to generate...",
                        lines=4,
                        value="Generate Q&A pairs about the provided documents."
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
                    logs_output = gr.Textbox(label="Server Logs", lines=10, max_lines=15, interactive=False, autoscroll=True)
                    start_logs_btn = gr.Button("Connect to Logs")
                    start_logs_btn.click(stream_logs, inputs=None, outputs=logs_output)

            generate_btn.click(
                fn=generate_data,
                inputs=[
                    prompt_input, files_upload, demo_upload, provider_dropdown, model_input, api_key_input,
                    temp_slider, count_slider,
                    agentic_checkbox, rag_checkbox, mcp_input,
                    conserve_chk, rate_slider,
                    hf_repo_input, hf_token_input, hf_private_chk, hf_append_chk, hf_config_dropdown, hf_split_dropdown
                ],
                outputs=[dataframe_output, json_output, status_output]
            )

        with gr.TabItem("Modify Dataset"):
            gr.Markdown("### 🛠 Modify Existing Hugging Face Dataset")
            with gr.Row():
                mod_repo = gr.Textbox(label="Repo ID")
                mod_token = gr.Textbox(label="Write Token", type="password")

            mod_fetch_btn = gr.Button("Load Dataset Info")
            with gr.Row():
                mod_config = gr.Dropdown(label="Config", allow_custom_value=True)
                mod_split = gr.Dropdown(label="Split", value="train", allow_custom_value=True)

            mod_fetch_btn.click(get_repo_info, inputs=[mod_repo, mod_token], outputs=[mod_config, mod_split])

            mod_op = gr.Radio(["append_rows", "add_column"], label="Operation", value="append_rows")

            with gr.Accordion("Generative Options", open=True):
                mod_gen_mode = gr.Checkbox(label="Use LLM Generation", value=False)
                with gr.Row():
                    mod_provider = gr.Dropdown(choices=PROVIDERS, value="groq", label="Provider")
                    mod_api_key = gr.Textbox(label="API Key", type="password")
                mod_instruction = gr.Textbox(label="Instruction (Prompt)", placeholder="Description of new column OR instruction for new rows")
                with gr.Row():
                    mod_new_col = gr.Textbox(label="New Column Name (for add_column)")
                    mod_rows = gr.Number(label="Number of Rows (for append_rows)", value=10)

            mod_data = gr.Textbox(label="Manual Data (JSON List of Dicts) - Only if NOT Generative", lines=5)

            mod_btn = gr.Button("Apply Modification")
            mod_status = gr.Textbox(label="Result")

            mod_btn.click(
                fn=modify_hf_dataset,
                inputs=[mod_repo, mod_token, mod_config, mod_split, mod_data, mod_op, mod_gen_mode, mod_instruction, mod_new_col, mod_rows, mod_provider, mod_api_key],
                outputs=[mod_status]
            )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
