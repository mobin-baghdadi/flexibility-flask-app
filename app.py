from flask import Flask, render_template, request, url_for
import cv2
import numpy as np
import os
import math

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    flexibility_degree = None
    error = None
    image_uploaded = False

    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            if image.filename != '':
                path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
                image.save(path)
                image_uploaded = True

        elif 'hip_x' in request.form:
            try:
                hip_x, hip_y = int(request.form['hip_x']), int(request.form['hip_y'])
                la_x, la_y = int(request.form['left_ankle_x']), int(request.form['left_ankle_y'])
                ra_x, ra_y = int(request.form['right_ankle_x']), int(request.form['right_ankle_y'])

                path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
                img = cv2.imread(path)
                if img is None:
                    error = "خطا در بارگذاری تصویر."
                    return render_template('index.html', error=error)

                joints = {
                    "main_hip": [hip_x, hip_y],
                    "left_ankle": [la_x, la_y],
                    "right_ankle": [ra_x, ra_y]
                }

                A = np.array(joints["left_ankle"], dtype=np.float32)
                B = np.array(joints["right_ankle"], dtype=np.float32)
                P = np.array(joints["main_hip"], dtype=np.float32)

                AB = B - A
                AP = P - A
                AB_norm = AB / np.linalg.norm(AB)
                proj_len = np.dot(AP, AB_norm)
                foot_proj = A + proj_len * AB_norm

                reference_length_pixels = np.linalg.norm(np.array([100, 100]) - np.array([100, 150]))

                def distance(p1, p2):
                    return np.linalg.norm(np.array(p1) - np.array(p2))

                a = distance(joints["left_ankle"], foot_proj) / reference_length_pixels
                b = distance(joints["right_ankle"], foot_proj) / reference_length_pixels
                h = distance(joints["main_hip"], foot_proj) / reference_length_pixels

                def is_super_flex():
                    mid_ankle_y = (joints["left_ankle"][1] + joints["right_ankle"][1]) / 2
                    hip_y = joints["main_hip"][1]
                    return hip_y > mid_ankle_y

                flexibility_degree = math.atan(a/h) + math.atan(b/h)
                flexibility_degree = math.degrees(flexibility_degree)
                if is_super_flex():
                    flexibility_degree = 360 - flexibility_degree

                image_uploaded = True  # to show image after angle is calculated

            except Exception as e:
                error = f"خطا در محاسبه: {e}"

    return render_template('index.html',
                           flexibility=flexibility_degree,
                           error=error,
                           image_uploaded=image_uploaded)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)