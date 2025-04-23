from flask import Flask, render_template, request, jsonify
from gradio_client import Client
from flask_cors import CORS 
import datetime
import shutil
import os

app = Flask(__name__)
CORS(app)

# Connect to Hugging Face Stable Diffusion space
client = Client("stabilityai/stable-diffusion")

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/list-static-images')
def list_static_images():
    import os
    static_dir = os.path.join(app.root_path, 'static')
    try:
        files = [f for f in os.listdir(static_dir) if os.path.isfile(os.path.join(static_dir, f))]
        return jsonify(files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/placeholder/<int:width>/<int:height>')
def placeholder(width, height):
    # You could return an actual placeholder image here
    return jsonify({
        "message": "Placeholder endpoint",
        "width": width,
        "height": height
    })

@app.route("/generate", methods=["POST"])
def generate():
    try:
        prompt = request.form.get("prompt", "A futuristic ")
        negative = "blurry, low quality, ugly"
        scale = 9
        api_name = "infer_2"
        api_dic=f"/{api_name}"

        # Generate image
        result = client.predict(
            prompt=prompt,
            negative=negative,
            scale=scale,
            api_name=api_dic
        )

        # result is a list of dicts with 'image' key

        if result and 'image' in result[0]:
            image_info = result[0]
            src_path = image_info['image']
        else:
            src_path = None
        temp = "anfas"

        # Save to static directory
        if not os.path.exists("static"):
            os.makedirs("static")

        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        filename = f"{api_name}_Img_{timestamp}.png"
        dst_path = os.path.join("static", filename)
        shutil.copy(src_path, dst_path)

        return render_template("generated.html", image_path=dst_path)

    except Exception as e:
        print("An error occurred:", e)
        return render_template("error.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True, port=8080)
