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

def generate_data(
    prompt,
    files,
    demo_file,
    provider,
    model,
    temperature,
    count,
    agentic,
    mcp_servers
):
    url = f"{API_URL}/generate"

    # Prepare form data
    data = {
        "prompt": prompt,
        "provider": provider,
        "temperature": float(temperature),
        "count": int(count),
        "agentic": agentic,
        "mcp_servers": mcp_servers if mcp_servers else None
    }

    if model:
        data["model"] = model

    files_payload = []
    if files:
        for f in files:
            # Gradio passes file paths
            files_payload.append(("files", (f.split("/")[-1], open(f, "rb"), "application/octet-stream")))

    if demo_file:
         files_payload.append(("demo_file", (demo_file.split("/")[-1], open(demo_file, "rb"), "text/csv")))

    try:
        response = requests.post(url, data=data, files=files_payload)
        response.raise_for_status()
        result = response.json()

        if result.get("success"):
            df = pd.DataFrame(result.get("data", []))
            json_output = json.dumps(result.get("data", []), indent=2)
            return df, json_output, "Success"
        else:
            return pd.DataFrame(), "{}", f"Error: {result}"

    except requests.exceptions.RequestException as e:
        error_detail = "Unknown error"
        if e.response is not None:
            try:
                error_detail = e.response.json().get("detail", e.response.text)
            except:
                error_detail = e.response.text
        return pd.DataFrame(), "{}", f"Failed: {error_detail}"
    except Exception as e:
         return pd.DataFrame(), "{}", f"Unexpected Error: {e}"

# --- UI Layout ---

with gr.Blocks(title="SLM Data Generator") as demo:
    gr.Markdown("# 🚀 Synthetic Data Generation Service")
    gr.Markdown("Generate high-quality datasets for SLM training/finetuning using various LLM providers.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Configuration")
            provider_dropdown = gr.Dropdown(choices=PROVIDERS, value="groq", label="LLM Provider")
            model_input = gr.Textbox(label="Model Name (Optional)", placeholder="e.g. llama3-8b-8192")
            temp_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, label="Temperature")
            count_slider = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="Number of Pairs")
            agentic_checkbox = gr.Checkbox(label="Enable Agentic Mode (Experimental)", value=False)
            mcp_input = gr.Textbox(label="MCP Servers (JSON List)", placeholder='["http://server1", "http://server2"]', lines=2)

            gr.Markdown("### 📄 Source Content")
            files_upload = gr.File(label="Upload Context Files (PDF, DOCX, CSV)", file_count="multiple")
            demo_upload = gr.File(label="Upload Few-Shot Demo (CSV)", file_count="single")

            generate_btn = gr.Button("✨ Generate Data", variant="primary", size="lg")

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

            # Streaming logs logic
            # Gradio doesn't support async generators directly in standard events easily without simple tricks.
            # But we can use a button to start streaming or use `demo.load`.
            # A common pattern is using a generator function.

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
            temp_slider,
            count_slider,
            agentic_checkbox,
            mcp_input
        ],
        outputs=[dataframe_output, json_output, status_output]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
