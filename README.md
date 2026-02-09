# SLM Data Generation Service

This service provides an API for generating synthetic data to train and fine-tune Small Language Models (SLMs). It includes endpoints for generating question-answer pairs from documents and for creating fine-tuning datasets based on user-provided prompts.

## Features

- **QA Generation:** Generate question-answer pairs from PDF or DOCX files.
- **Fine-tuning Data Generation:** Create custom fine-tuning datasets using a prompt and a specified model.

## API

### Generate Fine-tuning Data

- **Endpoint:** `POST /api/v1/generate-finetuning-data/`
- **Description:** Generates fine-tuning data based on a prompt and a model type.
- **Request Body:**
  ```json
  {
    "prompt": "Your data generation prompt",
    "model_type": "name-of-your-model"
  }
  ```
- **Response:**
  ```json
  {
    "success": true,
    "data": [
      {
        "generated_text": "Generated sample 1."
      },
      {
        "generated_text": "Generated sample 2."
      }
    ]
  }
  ```

## Getting Started

1.  **Install dependencies:**
    ```bash
    pip install -e .
    ```

2.  **Run the service:**
    ```bash
    uvicorn app.main:app --reload
    ```

The service will be available at `http://127.0.0.1:8000`.