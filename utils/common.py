import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric
import torchinfo
import torchvision
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


##### Get Device Details #####
def get_device() -> tuple:
    """Get Device type

    Returns:
        tuple: Device type
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return (use_cuda, device)


def get_batch_size(seed_value):

    # Set seed for reproducibility
    torch.manual_seed(seed_value)

    # Check if GPU/CUDA is available
    use_cuda, device = get_device()

    if use_cuda:
        torch.cuda.manual_seed(seed_value)

    if use_cuda:
        batch_size = 512
        #batch_size = 64
    else:
        batch_size = 64

    return batch_size

# def move_loss_to_cpu(loss):
#   moved_loss2cpu = [t.cpu().item() for t in loss]
#   return moved_loss2cpu

#####  Get the count of correct predictions
def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


test_incorrect_pred = {"images": [], "ground_truths": [], "predicted_vals": []}


#####  Get the incorrect predictions
def GetInCorrectPreds(pPrediction, pLabels):
    pPrediction = pPrediction.argmax(dim=1)
    indices = pPrediction.ne(pLabels).nonzero().reshape(-1).tolist()
    return indices, pPrediction[indices].tolist(), pLabels[indices].tolist()


def get_incorrect_test_predictions(model, device, test_loader):
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)

            ind, pred, truth = GetInCorrectPreds(output, target)
            test_incorrect_pred["images"] += data[ind]
            test_incorrect_pred["ground_truths"] += truth
            test_incorrect_pred["predicted_vals"] += pred

    return test_incorrect_pred

#####  Display the shape and decription of the train data
def display_train_data(train_data):

  print('[Train]')
  print(' - Numpy Shape:', train_data.cpu().numpy().shape)
  print(' - Tensor Shape:', train_data.size())
  print(' - min:', torch.min(train_data))
  print(' - max:', torch.max(train_data))
  print(' - mean:', torch.mean(train_data))
  print(' - std:', torch.std(train_data))
  print(' - var:', torch.var(train_data))


class model_accuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        preds = preds.argmax(dim=1)
        total = target.numel()
        self.correct += preds.eq(target).sum()
        self.total += total

    def compute(self):
        return 100 * self.correct.float() / self.total  


##### args: points should be of type list of tuples or lists
def plot_learning_rate_trend(curves,title,Figsize = (7,7)):
    fig = plt.figure(figsize=Figsize)
    ax = plt.subplot()
    for curve in curves:
        if("x" not in curve):
            ax.plot(curve["y"], label=curve.get("label", "label"))   
        else:
            ax.plot(curve["x"],curve["y"], label=curve.get("label","label"))
        plt.xlabel(curve.get("xlabel","x-axis"))
        plt.ylabel(curve.get("ylabel","y-axis"))
        plt.title(title)
    ax.legend()
    plt.show()

def save_cifar_image(array, path):
    # array is 3x32x32. cv2 needs 32x32x3
    # array = array.transpose(1,2,0)
    # # array is RGB. cv2 needs BGR
    # array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to PNG file
    return cv2.imwrite(path, array)

def visualize_dataset_images(display_image_list, 
                             display_label_list,
                             fig_title, 
                             figsize=None
                             ):
    fig = plt.figure(figsize=figsize)
    fig.suptitle(fig_title, fontsize=18)

    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.tight_layout()
        ele_img = display_image_list[i]
        plt.imshow(ele_img, cmap='gray')
        ele_label = display_label_list[i]
        plt.title(str(ele_label))
        plt.xticks([])
        plt.yticks([])

def get_gradcam_transform(model, img_tensor, denorm_img_tensor, pred_label, target_layer):

    # grad_cam = GradCAM(model=model, target_layers=[model.layer3[-1]],
    #                     use_cuda=(device == 'cuda'))
    _, device = get_device()
    grad_cam = GradCAM(model=model, target_layers=[target_layer],
                        use_cuda=(device == 'cuda'))    

    targets = [ClassifierOutputTarget(pred_label)]

    grayscale_cam = grad_cam(input_tensor=img_tensor.unsqueeze(0), targets=targets)
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]

    #grad_cam_output = show_cam_on_image(denorm_img_tensor, grayscale_cam, use_rgb=True, image_weight=0.7)
    grad_cam_output = show_cam_on_image(denorm_img_tensor, grayscale_cam, use_rgb=True, image_weight=0.60)

    return grad_cam_output