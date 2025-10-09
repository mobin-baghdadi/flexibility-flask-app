from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import math
import re

app = Flask(__name__)

def decode_base64_image(base64_str):
    # تبدیل base64 به آرایه numpy
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    img_data = base64.b64decode(base64_data)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    try:
        img_base64 = data['image']
        joints = data['joints']  # dict: {"hip": [x,y], "left_ankle": [x,y], "right_ankle": [x,y]}

        img = decode_base64_image(img_base64)
        if img is None:
            return jsonify({'error': 'خطا در بارگذاری تصویر'}), 400

        hip = joints['hip']
        left_ankle = joints['left_ankle']
        right_ankle = joints['right_ankle']

        A = np.array(left_ankle, dtype=np.float32)
        B = np.array(right_ankle, dtype=np.float32)
        P = np.array(hip, dtype=np.float32)

        AB = B - A
        AP = P - A
        AB_norm = AB / np.linalg.norm(AB)
        proj_len = np.dot(AP, AB_norm)
        foot_proj = A + proj_len * AB_norm

        reference_length_pixels = np.linalg.norm(np.array([100, 100]) - np.array([100, 150]))

        def distance(p1, p2):
            return np.linalg.norm(np.array(p1) - np.array(p2))

        a = distance(left_ankle, foot_proj) / reference_length_pixels
        b = distance(right_ankle, foot_proj) / reference_length_pixels
        h = distance(hip, foot_proj) / reference_length_pixels

        mid_ankle_y = (left_ankle[1] + right_ankle[1]) / 2
        super_flex = hip[1] > mid_ankle_y

        flexibility_degree = math.degrees(math.atan(a/h) + math.atan(b/h))
        if super_flex:
            flexibility_degree = 360 - flexibility_degree

        return jsonify({'flexibility': round(flexibility_degree, 2)})

    except Exception as e:
        return jsonify({'error': f'خطا در محاسبه: {e}'}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
