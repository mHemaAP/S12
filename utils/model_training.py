import os
import pandas as pd
import torch
#import torchvision
from torchvision.utils import save_image
from collections import defaultdict
import matplotlib.pyplot as plt
from torch_lr_finder import LRFinder
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from torchvision.utils import save_image
# from datasets.cifar10_dataset import cifar10Set
# from models.custom_Resnet import custResNet
from utils.common import *

# torch.manual_seed(11)

class trainModel():
    """ Class to encapsulate model training """

    def __init__(self, model, dropout=0.01,
                 batch_size=32,
                 epochs=None,
                 precision="32-true"
                 ):

        self.model = model
        self.dataset = model.dataset       
    
        self.epochs = epochs

        self.checkpoint_callback = ModelCheckpoint( dirpath=os.getcwd(),
                                                    #filename="{epoch}-{model.validation_accuracy:.3f}-{model.train_accuracy:.3f}",
                                                    filename="{epoch}-{model.validation_step_loss:.3f}",
                                                    save_top_k=1,
                                                    verbose=True,
                                                    monitor='validation_step_loss',
                                                    mode='min'
                                                )
        self.model_summary_callback = ModelSummary(max_depth=10)


        self.trainer = Trainer(callbacks=[self.checkpoint_callback, 
                                             self.model_summary_callback],
                                  max_epochs=epochs or model.epochs,
                                  precision=precision)
            
        self.incorrect_preds = None
        self.misclass_preds_df = None
        self.file_no = 1        


    def run_training_model(self):
        self.trainer.fit(self.model)


    ### Display the model statistics         
    def display_model_stats(self):
        fig, axs = plt.subplots(2,2,figsize=(15,10))
        train_loss = [t.cpu().item() for t in self.train_loss]
        axs[0, 0].plot(train_loss)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(self.train_accuracy)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(self.test_loss)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(self.test_accuracy)
        axs[1, 1].set_title("Test Accuracy")

    def get_misclassified_predictions(self):

        self.incorrect_preds = defaultdict(list)
        incorrect_images = list()
        processed = 0
        prediction_results = self.trainer.predict()

        for (data, target), pred in zip(self.model.predict_dataloader(), prediction_results):
            ind, pred, truth = GetInCorrectPreds(pred, target)
            self.incorrect_preds["indices"] += [x + processed for x in ind]
            incorrect_images += data[ind]
            self.incorrect_preds["ground_truths"] += truth
            self.incorrect_preds["predicted_vals"] += pred
            processed += len(data)

        self.misclass_preds_df = pd.DataFrame(self.incorrect_preds)
        self.incorrect_preds["images"] = incorrect_images
        self.misclass_preds_df.to_csv('misclassified_images.csv')


    ### Display the incorrect predictions of the CIFAR10 data images
    def show_cifar10_incorrect_predictions(self, figsize=None, 
                                           denormalize=True, enable_grad_cam=False, 
                                           target_layer=None):
        if self.incorrect_preds is None:
            self.get_misclassified_predictions()

        images_list = []
        incorrect_pred_labels_list = []

        for i in range(20):
            img = self.incorrect_preds["images"][i]
            pred = self.incorrect_preds["predicted_vals"][i]
            gtruth = self.incorrect_preds["ground_truths"][i]

            if (enable_grad_cam==True):
                denorm_img = self.dataset.inv_transform_image(img).cpu().numpy()
                

                img = get_gradcam_transform(self.model, img, 
                                      denorm_img, 
                                      pred, target_layer)                
            elif(denormalize==True):
                img = self.dataset.inv_transform_image(img).cpu()

            if self.dataset.classes is not None:
                pred = f'P{pred}:{self.dataset.classes[pred]}'
                gtruth = f'A{gtruth}:{self.dataset.classes[gtruth]}'
            
            label = f'{pred}::{gtruth}'

            images_list.append(img)
            incorrect_pred_labels_list.append(label)

        visualize_dataset_images(images_list,
                                 incorrect_pred_labels_list,
                                 ' Mis-Classified Images - Predicted Vs Actual ',
                                 figsize=(10, 8)
                                 )

    def save_misclassified_to_file(self):

        path = './misclassifed_examples'
        os.mkdir(path)
        os.chdir(path)
        for i in range(len(self.incorrect_preds["images"])):
            img = self.incorrect_preds["images"][i]
            save_image(img, '{}.jpg'.format(i), normalize=True)        
       
    def save_model(self):
        torch.save(self.model.state_dict(), 'cust_resnet_model_saved.pth')

