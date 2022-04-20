import tensorflow as tf
import numpy as np
from PIL import Image
from keras.preprocessing import image

model = tf.keras.models.load_model('translearnresnetmodel.h5')
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

def preprocess_image(img, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize(target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255
    return img

def predict(data):
    img = Image.open(data)
    processed_image = preprocess_image(img, target_size=(150,150))

    images = np.vstack([processed_image])
    classes = model.predict(images, batch_size = 32)
    #label = np.where(classes[0] > 0.5, 1, 0)
    if classes[0][0] >= .80:
        conf = round(float(classes[0])*100, 2)
        return f"Fresh Apple ({conf}%)"
    elif classes[0][1] >= .80:
        conf = round(float(classes[0])*100, 2)
        return f"Fresh Banana ({conf}%)"
    elif classes[0][2] >= .80:
        conf = round(float(classes[0])*100, 2)
        return f"Fresh Bitter Gourd ({conf}%)"
    elif classes[0][3] >= .80:
        conf = round(float(classes[0])*100, 2)
        return f"Fresh Capsicum ({conf}%)"
    elif classes[0][4] >= .80:
        conf = round(float(classes[0])*100, 2)
        return f"Fresh Orange ({conf}%)"
    elif classes[0][5] >= .80:
        conf = round(float(classes[0])*100, 2)
        return f"Fresh Tomato ({conf}%)"

    elif classes[0][6] >= .80:
        conf = round(float(classes[0])*100, 2)
        return f"Rotten Apple ({conf}%)"
    elif classes[0][7] >= .80:
        conf = round(float(classes[0])*100, 2)
        return f"Rotten Banana ({conf}%)"
    elif classes[0][8] >= .80:
        conf = round(float(classes[0])*100, 2)
        return f"Rotten Bitter Gourd ({conf}%)"
    elif classes[0][9] >= .80:
        conf = round(float(classes[0])*100, 2)
        return f"Rotten Capsicum ({conf}%)"
    elif classes[0][10] >= .80:
        conf = round(float(classes[0])*100, 2)
        return f"Rotten Orange ({conf}%)"
    elif classes[0][11] >= .80:
        conf = round(float(classes[0])*100, 2)
        return f"Rotten Tomato ({conf}%)"
        
    else:
        return "Not Sure"