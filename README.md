# ComfyUI Text-to-Image API

This repository provides a simple FastAPI wrapper for ComfyUI to expose a text-to-image generation endpoint.

## Purpose

The main goal of this project is to provide a straightforward way to use ComfyUI for text-to-image generation through a simple API. This allows for easy integration with other services and applications.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run ComfyUI:**
    Ensure that you have a running instance of ComfyUI. By default, the API connects to `http://127.0.0.1:8188`.

3.  **Run the FastAPI Server:**
    ```bash
    uvicorn main:app --reload
    ```

## Usage

Once the server is running, you can use the `/txt2img` endpoint to generate images.

### Endpoint: `/txt2img`

*   **Method:** `POST`
*   **Description:** Generates an image from a text prompt.
*   **Query Parameters:**
    *   `prompt` (string, required): The text prompt to generate the image from.
*   **Example:**
    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/txt2img?prompt=a%20beautiful%20landscape' \
      -H 'accept: application/json' \
      -d ''
    ```

The API will return the generated image as a PNG file.
