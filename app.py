from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import subprocess  # For ffmpeg
import numpy as np 

app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')
processed_folder = os.path.join('static', 'processed')

app.config['UPLOAD_FOLDER'] = upload_folder
app.config['PROCESSED_FOLDER'] = processed_folder

def region_of_interest(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    polygons = np.array([[(0, height-5), (width, height-5), (356, 222), (417,222)]])
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, polygons,255)
    masked_image = cv2.bitwise_and(frame, mask)
    return masked_image


def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Adjust codec if needed
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame here (replace with your desired logic)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 75, 150)
        cropped_frame = region_of_interest(edges)
        lines = cv2.HoughLinesP(cropped_frame, 1, np.pi / 180, 50, maxLineGap=10000000)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['videoFile']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process video and generate output filename
        processed_filename = f"processed_{filename}"  # Adjust naming convention
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        process_video(filepath, processed_path)

        # Return URL for processed video
        return render_template('image_render.html', processed_video=os.path.join(app.config['PROCESSED_FOLDER'], processed_filename))
    return render_template('image_render.html')

if __name__ == '__main__':
    app.run(debug=True, port=8001)
