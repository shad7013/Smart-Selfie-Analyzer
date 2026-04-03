import torch 
import cv2
import numpy as np 


class GradCAM:
  def __init__(self, model, target_layer):
    self.model = model
    self.target_layer = target_layer

    self.gradients = None 
    self.activations = None

    # Hook for gradients 
    target_layer.register_forward_hook(self.save_gradients)

    # Hook for activations 
    target_layer.register_forward_hook(self.save_activations)

  def save_gradients(self, module, grad_input, grad_output):
    self.gradients = grad_output[0]

  def save_activations(self, module, input, output):
    self.activations = output 

  def generate(self, input_tensor, target_class, task="emotion"):

    self.model.zero_grad()

    outputs = self.model(input_tensor)
    if isinstance(outputs, dict):
      output = outputs[task]
    else:
      raise ValueError("Model output should be a dictionary.")

    loss = output[:, target_class]
    loss.backward()

    gradients = self.gradients
    activations = self.activations

    # Ensure correct shapes
    if len(activations.shape) == 4:
        activations = activations[0]   # (C, H, W)

    if len(gradients.shape) == 4:
        gradients = gradients[0]       # (C, H, W)

    # Now both are (C, H, W)
    weights = torch.mean(gradients, dim=(1, 2))

    cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

    for i, w in enumerate(weights):
        cam += w * activations[i]

    cam = torch.relu(cam)
    cam = cam.detach().cpu().numpy()
    
    # resize CAM to input image size
    cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))

    # normalize CAM to [0, 1]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


    return cam 
  
def overlay_heatmap(image, cam):

  image = np.array(image)

  heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
  heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

  overlay = heatmap * 0.4 + image * 0.6
  overlay = np.uint8(overlay)

  return overlay 

