# ComfyUI + CogVideoX Video Generation Pipeline

This repository provides a Dockerized pipeline for generating short videos from face images using the **CogVideoX-5b** model hosted on Hugging Face. The workflow takes a face image, generates a descriptive text prompt, sends the request to the Hugging Face API, and returns both a video (MP4) and a thumbnail image.

## Features
- **Automated Prompt Generation**: Creates descriptive text prompts from input face images.  
- **Video Generation**: Utilizes the Hugging Face-hosted [THUDM/CogVideoX-5b](https://huggingface.co/THUDM/CogVideoX-5b) model for text-to-video synthesis.  
- **End-to-End Workflow**: Fully Dockerized environment for easy setup and reproducibility.  
- **Outputs**: Returns a final video clip along with a thumbnail image.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. Build the Docker image:
   ```bash
   docker build -t cogvideo-pipeline .
   ```

3. Run the container:
   ```bash
   docker run -it --rm \
     -v $(pwd)/faces:/app/faces \
     -e HF_TOKEN=your_huggingface_token \
     cogvideo-pipeline
   ```

## Usage

Run the script inside the container:
```bash
python script.py --artist faces/example.jpg
```

This will:
1. Generate a descriptive text prompt from the provided face image.  
2. Send the prompt and parameters to the Hugging Face API.  
3. Save the generated video and thumbnail in the `output/` directory.

## Repository Structure
```
faces/              # Input face images
script.py           # Main pipeline script
requirements.txt    # Python dependencies
Dockerfile          # Docker container setup
README.md           # Documentation
```

## Requirements
- Docker (recommended)  
- Hugging Face account and API token (`HF_TOKEN`)  
- Alternatively, run locally with:
  ```bash
  pip install -r requirements.txt
  ```

## Links
- CogVideoX-5b Model: [https://huggingface.co/THUDM/CogVideoX-5b]
- Hugging Face Space [https://huggingface.co/spaces/yijin928/Test]
- Video Demo [https://youtu.be/cy2CbJM4DY0]

© 2025 musicnbrain.org. All rights reserved.
