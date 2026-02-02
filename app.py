from flask import Flask, render_template, request
import torch
from PIL import Image
import os
import cv2
import io
import base64
import shutil

app = Flask(__name__)

# Configure Torch to use a writable directory for caching
try:
    torch.hub.set_dir('/tmp/torch_hub')
except Exception:
    pass # valid on some systems, might fail on others, but worth trying

# Load your vehicle detection model
# on Vercel, we need to ensure we can download the hub repo. 
# If it fails, we might need a local copy of yolov5, but let's try standard first.
model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/bestf.pt', force_reload=True) 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Process based on file type
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            # Read file to memory
            file_bytes = file.read()
            
            # Prepare uploaded image for display (Base64)
            upload_b64 = "data:image/jpeg;base64," + base64.b64encode(file_bytes).decode('utf-8')
            
            # Open image for processing
            image = Image.open(io.BytesIO(file_bytes))
            
            # Run detection
            results = model(image)
            
            # Render result -> returns list of numpy arrays
            rendered_frames = results.render() 
            
            # Convert first frame back to PIL Image and then Base64
            if rendered_frames and len(rendered_frames) > 0:
                result_pil = Image.fromarray(rendered_frames[0])
                
                # Save to buffer
                buf = io.BytesIO()
                result_pil.save(buf, format="JPEG")
                process_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')
                
                return render_template('index.html', uploaded_file=upload_b64, processed_file=process_b64, file_type='image')
            else:
                return "Model failed to render output", 500
                
        except Exception as e:
            return f"Error processing image: {str(e)}", 500

    elif file.filename.lower().endswith(('.mp4', '.avi')):
        return "Video processing is not supported in this serverless deployment due to resource limits.", 400
    else:
        return "Unsupported file format", 400

# Video processing functions removed as they are not viable for serverless

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
