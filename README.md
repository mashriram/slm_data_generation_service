# SLM Data Generation Service

A comprehensive service to generate high-quality synthetic data for Small Language Models (SLMs) training and fine-tuning.
It supports multiple LLM providers, file context (PDF, DOCX, CSV), few-shot examples via CSV, and agentic behavior.

## Features

- **Unified Generation API**: Generate data from prompts, documents, or hybrid approaches.
- **Multi-Provider Support**: Groq, OpenAI, Google Gemini, HuggingFace (local).
- **File Support**: PDF, DOCX, CSV (as context or few-shot examples).
- **Agentic Mode**: Experimental mode to let an agent orchestrate the generation process.
- **Real-time Logs**: WebSocket endpoint `/ws/logs` for streaming server logs.
- **Gradio UI**: User-friendly interface to interact with the service.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repo-url>
    cd slm-data-generation-service
    ```

2.  **Install dependencies**:
    ```bash
    pip install .
    pip install gradio  # For the UI
    ```

3.  **Environment Variables**:
    Create a `.env` file in the root directory:
    ```env
    GROQ_API_KEY=your_groq_key
    OPENAI_API_KEY=your_openai_key
    GOOGLE_API_KEY=your_google_key
    ```

## Usage

### Start the Backend Server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.
Swagger Docs: `http://localhost:8000/docs`

### Start the Gradio UI

In a separate terminal:

```bash
python gradio_app.py
```

Access the UI at `http://localhost:7860`.

## API Endpoints

-   `POST /api/v1/generate`: Generate data.
-   `GET /api/v1/models`: List available models.
-   `WS /ws/logs`: Stream logs.

## Testing

Run the included tests:

```bash
pytest
```
