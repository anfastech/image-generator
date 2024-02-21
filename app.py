from flask import Flask, render_template, request
import shutil
from gradio_client import Client
import datetime

app = Flask(__name__)

# Set timeout for HTTP requests
client = Client("prodia/fast-stable-diffusion")
negative_prompt = "3d, cartoon, anime, eformed eyes, deformed nose, deformed ears, deformed nose, bad anatomy, ugly, extra limb, low resolution, pixelated, distorted proportions, unnatural colors, unrealistic lighting, excessive noise, unrealistic shadows, unnatural poses"

stable_diffusion_checkpoint = "amIReal_V41.safetensors [0a8a2e61]"
sampling_steps = 20.0 
sampling_method = "DPM++ 2M Karras"
cfg_scale = 7.0
width = 512
height = 512
seed = -1.0


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        # prompt = request.form["prompt"]
        prompt = "Design a minimalist sitting room that exudes tranquility and elegance. Incorporate room decor items that are visually captivating and evoke a sense of emotional resonance. Your design should prioritize simplicity, clean lines, and a harmonious color palette. Consider how each element contributes to creating a serene and inviting atmosphere for relaxation and contemplation"

        # Generate image using your code
        result = client.predict(
            prompt,
            negative_prompt,
            stable_diffusion_checkpoint,
            sampling_steps,
            sampling_method,
            cfg_scale,
            width,
            height,
            seed,
            api_name="/txt2img"
        )

        # Save image
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        destination_path = f"static/Img_{timestamp}.png"  # Save to static directory
        shutil.move(result, destination_path)

        return render_template("generated.html", image_path=destination_path)

    except Exception as e:
        # Handle Gradio API errors or other exceptions
        error_message = f"An error occurred: {str(e)}"
        return render_template("error.html", error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True, port=8080)