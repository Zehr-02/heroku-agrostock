import tensorflow as tf
import numpy as np
from PIL import Image
from keras.preprocessing import image

model = tf.keras.models.load_model('translearnresnetmodel.h5')

def preprocess_image(img, target_size):
  if img.mode != "RGB":
      img = img.convert("RGB")

  img = img.resize(target_size)
  img = image.img_to_array(img)
  img = img.reshape(150, 150, 3)
  img = np.expand_dims(img, axis=0)
  img = tf.keras.applications.resnet50.preprocess_input(img)
  return img

def predict(data):
  img = Image.open(data)
  processed_image = preprocess_image(img, target_size=(150,150))

  images = np.vstack([processed_image])
  classes = model.predict(images, batch_size=10)

  #return f"output = ({classes}%"
    #label = np.where(classes[0] > 0.5, 1, 0)

  class_names = ['Fresh Apple', 'Fresh Banana', 'Fresh Bitter Gourd', 'Fresh Capsicum', 'Fresh Orange', 'Fresh Tomato', 'Rotten Apple', 'Rotten Banana', 'Rotten Bitter Gourd', 'Rotten Capsicum', 'Rotten Orange', 'Rotten Tomato']
  predict_result = class_names[np.argmax(classes)]
  #confidence_value = np.argmax(classes)

  return f"{predict_result}"

  #return f"{predict_result} ({confidence_value}),\n \n {classes} "
  """
  if model.predict(images)[0][0]>= .89:
        #conf = round(float(classes[0])*100, 2)
        return f"Fresh Apple"#({conf}%)"
  elif model.predict(images)[0][1]>= .89:
        #conf = round(float(classes[0])*100, 2)
        return f"Fresh Banana"#({conf}%)"
  elif model.predict(images)[0][2]>= .89:
       # conf = round(float(classes[0])*100, 2)
        return f"Fresh Bitter Gourd"#({conf}%)"
  elif model.predict(images)[0][3]>= .89:
      #  conf = round(float(classes[0])*100, 2)
        return f"Fresh Capsicum"#({conf}%)"
  elif model.predict(images)[0][4]>= .89:
      #  conf = round(float(classes[0])*100, 2)
        return f"Fresh Orange"#({conf}%)"
  elif model.predict(images)[0][5]>= .89:
      #  conf = round(float(classes[0])*100, 2)
        return f"Fresh Tomato"#({conf}%)"

  elif model.predict(images)[0][6]>= .89:
      #  conf = round(float(classes[0])*100, 2)
        return f"Rotten Apple"#({conf}%)"
  elif model.predict(images)[0][7]>= .89:
     #  conf = round(float(classes[0])*100, 2)
        return f"Rotten Banana"#({conf}%)"
  elif model.predict(images)[0][8]>= .89:
     #   conf = round(float(classes[0])*100, 2)
        return f"Rotten Bitter Gourd"#({conf}%)"
  elif model.predict(images)[0][9]>= .89:
      #  conf = round(float(classes[0])*100, 2)
        return f"Rotten Capsicum"#({conf}%)"
  elif model.predict(images)[0][10]>= .89:
      #  conf = round(float(classes[0])*100, 2)
        return f"Rotten Orange"#({conf}%)"
  elif model.predict(images)[0][11]>= .89:
       # conf = round(float(classes[0])*100, 2)
        return f"Rotten Tomato"#({conf}%)"
        
  else:
        return "Not Sure"
    """
