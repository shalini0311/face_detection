import os
import cv2
import numpy
from PIL import Image
from StringIO import StringIO
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Get port from environment variable or choose 9099 as local default
port = int(os.getenv("PORT", 9099))

size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = r'database'
model = cv2.face.LBPHFaceRecognizer_create()
(width, height) = (112, 92)
face_cascade = cv2.CascadeClassifier(haar_file)
(images, lables, names, id) = ([], [], {}, 0)


def training_dataset():
    print('Training...')
    # Create a list of images and a list of corresponding names

    for (subdirs, dirs, files) in os.walk(datasets):
        for subdir in dirs:
            global id
            names[id] = subdir
            subjectpath = os.path.join(datasets, subdir)
            for filename in os.listdir(subjectpath):
                path = subjectpath + '/' + filename
                lable = id
                global images
                global lables
                images.append(cv2.imread(path, 0))
                lables.append(int(lable))
            id += 1

    # (width, height) = (130, 100)
    # Create a Numpy array from the two lists above
    (images, lables) = [numpy.array(lis) for lis in [images, lables]]

    # OpenCV trains a model from the images
    # NOTE FOR OpenCV2: remove '.face'
    model.train(images, lables)


def recognise_faces(faces, open_cv_image):
    person_detected = 'Unknown'
    for (x, y, w, h) in faces:
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2GRAY)
        cv2.rectangle(open_cv_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        # Try to recognize the face
        prediction = model.predict(face_resize)
        cv2.rectangle(open_cv_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        print('prediction is: ', prediction)
        if prediction[1] < 120:
            name = names[prediction[0]]
        else:
            name = 'Unknown'
        return name

def detect_faces(image):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    img = Image.open(StringIO(image))
    img_cv2 = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2RGBA)
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_RGBA2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    try:
        name = recognise_faces(faces, img_cv2)
        return [name, faces.tolist()]
    except:
        return ['',[]]


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def prediction():
    """
    curl -X POST -v -H "Content-Type: image/png" --data-binary @abba.png http://127.0.0.1:9099/prediction -o foo.jpg
    """
    if request.method == "POST":
        image = request.data
        name, face_coordinates = detect_faces(image)
        print 'Detected person is: ' + str(name)
        return jsonify(faces=face_coordinates, name = name)


if __name__ == "__main__":
    training_dataset()
    app.run(port=port)

