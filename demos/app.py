
# 1. imports and class names setup
import gradio as gr
import os
import torch
import torchvision


from model import create_resnet50_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
with open("class_names.txt", "r") as f:
  class_names = [pet.strip() for pet in f.readlines()]

# 2. model and transforms preparation
resnet50_pets, resnet50_transforms = create_resnet50_model(num_classes=37)

# Load saved weights
resnet50_pets.load_state_dict(
    torch.load(
        f="pretrained_resnet50_pets.pth",
        map_location=torch.device("cpu") # load the model to the CPU
    )
)

from typing import Tuple, Dict
def predict(img) -> Tuple[Dict, float]:
  # Start a timer
  start_time = timer()

  # Transform the input image for use with EffNetB2
  img = resnet50_transforms(img).unsqueeze(0) # unsqueeze = add batch dimension on 0th index

  # Put model into eval mode, make prediction
  resnet50_pets.eval()
  with torch.inference_mode():
    # Pass transformed image through the model and turn the prediction logits into probaiblities
    pred_probs = torch.softmax(resnet50_pets(img), dim=1)

  # Create a prediction label and prediction probability dictionary
  # ความน่าจะเป็นการทำนาย (prediction probability)
  pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

  # Calculate pred time
  end_time = timer()
  pred_time = round(end_time - start_time, 4)

  # Return pred dict and pred time
  return pred_labels_and_probs, pred_time


#create a example list
example_list = ["examples/" + example for example in os.listdir("examples")]
example_list


title = "Resnet50 Pet 🐶🐱🐈"
description = "An Resnet50 feature extractor computer vision model to classify Pet images into 37 classes Dog & Cat"
article = " Created at [https://github.com/raksitkan/Pytorch_vision_Pet])."
demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)
demo.launch()
