# machineLearning-visually-impaired-
A ML project on visually impaired

Project Idea: Object Recognition for Assistive Apps for Blind Users using the VizWiz Classification Dataset.

Description: Modern object recognition models are trained on clean datasets like ImageNet, but visually impaired users take pictures that are often blurry, poorly framed or low lighted. The goal is to classify objects in bad images to power an assistive app that could speak out what's in front of the user. We will use the VizWiz classification dataset which contains images taken by blind people and labeled with object categories.

Dataset: VizWiz Classification (https://vizwiz.org)
Images taken by blind users, labeled with presence/absence of 200 ImageNet style object cateogries, designed to test classifiers trained on ImageNet. 
Scale: about 8,900 images (test set) plus associated metadata and category list
This is perfect because:
It's real data from visually impaired users not synthetic
Difficult: blue, off center subjects, bad lighting are good for analysis
Directly tied to assistive tech: captioning/recognition for blind people

Problem Formulation:
Input: An RGB image x ∈ R^(H×W×3) captured by a blind user
Output: A probability distribution over K object categories, p(y|x) where y ∈ {1,...,K}. We will assume it's a single-label classification even though some images may contain multiple objects. 
Training data: {(xi​,yi​)}i=1-N​ whre xi is an image and yi is the ground truth label.
Preprocessing: Resize to 224x224. Normalize with ImageNet means/stds

Loss function: Cross entropy loss for multi class classification

Training procedure: 

Optimizer: Adam or SGD with momentum

Learning rate schedule: step decay or cosine schedule

Epochs: 10-30 with early stopping on validation loss

Batch size: 32-128

ML approaches to compare:

1. Model 1-Baseline: Linear classifier on frozen features
use a pretrained CNN
freeze all convolutional layers
extract features from the penultimate layer, then train:
logistic regression or
linear SVM
this gives us a simple baseline and separation between "feature extraction" and "classifier"

2. Model 2-fine tuned CNN
start from a pretrained ResNet-50
replace the final fully connected layer with a K class output
Fine tune:
   first, train only the final layer
   then optionally unfreeze last 1-2 blocks and continue training with a smaller LR.
Regularization: data augmentation to simulate real-world variation

3. Model 3: Vision Transformer or CLIP based approach
Pick one:
Option A: ViT (vision transformer)
   use vit_b_16 or similar from timm/torchvision
   fine tune same way as Model 2
Option B: CLIP zero shot vs CLIP fine tuned
   Use CLIP to generate text prompts like "a photo of a {category}"
   Compute similarity between image embedding and text embeddings to do zero shot classification
   then optionally fine tune a lightweight classifier on top of CLIP's image embeddings

 **Evaluation:** Models are evaluated using top-1 and top-5 accuracy, macro-F1, confusion matrices, and qualitative error analysis. We discuss how misclassifications might affect visually impaired users in real assistive scenarios.
 
Evaluation: We report the following metrics on the held out teset set:

Top 1 accuracy

Top 5 accuracy

Macro F1 score

per class accuracy

confusion matrix visualizations

Also include qualitative examples of:

Correct predictions on difficult images

failure cases that would be problematic for visually impaired users

**Outcome:** Fine-tuning a modern CNN significantly improves accuracy over the baseline, but the dataset remains challenging, highlighting gaps between standard benchmarks and accessibility-focused applications.

| Model                        | Top-1 Acc | Top-5 Acc | Macro-F1 |
| ---------------------------- | --------- | --------- | -------- |
| Linear on ResNet-18 features | TODO      | TODO      | TODO     |
| Fine-tuned ResNet-50         | TODO      | TODO      | TODO     |
| ViT / CLIP (fine-tuned)      | TODO      | TODO      | TODO     |

How to Run

Download the dataset and place it under data/.

Install dependencies: pip install -r requirements.txt

Train a model: python train.py ...

Evaluate: python eval.py ...
