from flask import Flask, render_template, request
from gradio_client import Client
import datetime
import shutil
import os

app = Flask(__name__)

# Connect to Hugging Face Stable Diffusion space
client = Client("stabilityai/stable-diffusion")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        prompt = request.form.get("prompt", "A futuristic city at sunset with flying cars.")
        negative = "blurry, low quality, ugly"
        scale = 9

        # Generate image
        result = client.predict(
            prompt=prompt,
            negative=negative,
            scale=scale,
            api_name="/infer"
        )

        # result is a list of dicts with 'image' key
        image_info = result[0]  # Just get the first image
        src_path = image_info['image']

        # Save to static directory
        if not os.path.exists("static"):
            os.makedirs("static")

        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        filename = f"Img_{timestamp}.png"
        dst_path = os.path.join("static", filename)
        shutil.copy(src_path, dst_path)

        return render_template("generated.html", image_path=dst_path)

    except Exception as e:
        print("An error occurred:", e)
        return render_template("error.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True, port=8080)
