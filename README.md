# Resnet

## Problem Statement

In the context of the final project, the challenge is to apply transfer learning techniques to train a classifier for a complex dataset that combines the Stanford Cars, Food101, Flowers, and FGVC Aircraft datasets. The objective is to achieve accurate classification for a subset of 100 classes, each containing 10 training images. The datasets present varying characteristics, such as different image dimensions and levels of noise.

To address this task, participants are required to:

Utilize transfer learning by loading pre-trained model weights and fine-tuning them on the provided dataset.
Implement advanced architectures and employ sophisticated data augmentations for improved performance.
Resize all images to a consistent size (e.g., 224x224) before training.
Ensure reproducibility by using a manual seed, repeating experiments, and reporting averaged training and validation accuracies/losses.
Provide detailed instructions, code, and model weights on Canvas for reproducibility.
The complexity lies in effectively adapting and fine-tuning pre-trained models to achieve high accuracy on a challenging dataset with limited samples. Participants must balance the use of modern, computationally intensive models with the dataset's relatively small size to prevent overfitting. 

## Solution
. Models Used:
The solution explores various pre-trained models for transfer learning, including:

Resnet50
Resnet50V2
Resnet101
Resnet101V2
Resnet152
VGG16
VGG19
Xception
InceptionResnetv2
EfficientnetBo
Swintransformer
ViT
2. Evaluation of Models:
Models that did not work well:

a. Resnet50V2:

Reasons for Failure:
No significant improvement over Resnet50.
EMA did not yield positive results.
Computation limitations hindered extensive experimentation.
b. Resnet101:

Reasons for Limited Usage:
Marginal improvement over Resnet50.
Increased complexity without substantial performance gain.
Simplified models were preferred due to computational constraints.
c. Resnet101V2:

Reasons for Failure:
No improvement observed, and performance deteriorated.
Increased complexity did not justify the results.
Computational overhead without tangible benefits.
d. VGG16 and VGG19:

Reasons for Failure:
Did not achieve desired accuracy.
Deep architectures might have led to overfitting.
Computationally expensive for minimal gain.
e. Xception:

Reasons for Failure:
Despite various hyperparameter combinations, accuracy remained suboptimal.
Computationally expensive with no clear advantage.
Challenging convergence during training.
f. InceptionResnetV2, EfficientnetB0, Swintransformer, ViT:

Reasons for Limited Usage:
Implementation issues (EfficientnetBo, Swintransformer).
Accuracy drop after a certain number of epochs (ViT).
Overall complexity and resource requirements.
Tradeoffs of These Models:

Computation vs. Performance: More complex models demand higher computational resources, and the gain in accuracy may not be proportionate.
Overfitting vs. Generalization: Deeper architectures increase the risk of overfitting, especially in datasets with limited samples.
Resource Constraints: Some models had to be excluded due to implementation challenges and resource limitations.
3. Successful Model: ResNet50
Model Configuration:

The code utilizes ResNet50 for transfer learning, freezing the initial layers and fine-tuning the model for the specific dataset. Key steps include:

Data Loading and Preprocessing:

Images are loaded, resized, and preprocessed using the preprocess_input function.
Training and validation datasets are split.
Data Augmentation:

ImageDataGenerator is used for augmentation, including rotation, shifting, shearing, zooming, flipping, and brightness adjustments.
Advanced data augmentation techniques like mixup and random_eraser are employed.
Label Encoding and Mixup Generator:

Label encoding is applied to target labels.
Mixup generator is utilized to enhance the model's robustness.
Model Creation and Compilation:

ResNet50 is loaded with pre-trained weights, and the top layers are fine-tuned.
A global average pooling layer, dense layers, and dropout are added.
The model is compiled using categorical crossentropy loss and the Adam optimizer.
Training:

The model is trained using a generator with mixup augmentation.
The training history is collected and stored in a CSV file.
Evaluation:

The model is evaluated on the validation dataset, and accuracy is calculated using the predicted classes.
Model Saving:

The trained model is saved in the "model.h5" file for future use.
Significance and Reasons for Success:

Transfer Learning with ResNet50:

ResNet50 is known for its effectiveness in image classification tasks.
Fine-tuning allows leveraging pre-trained features, adapting the model to the specifics of the combined dataset.
Freezing Initial Layers:

Freezing the initial layers prevents the loss of valuable pre-trained features.
Fine-tuning the later layers helps capture dataset-specific patterns.
Data Augmentation and Mixup:

Augmenting the dataset with various transformations improves model generalization.
Mixup generator enhances robustness by blending images and their corresponding labels.
Regularization:

Dropout is applied for regularization, reducing overfitting during training.
L2 regularization in the dense layer further aids in preventing overfitting.
Adam Optimizer:

Adam optimizer adapts learning rates during training, speeding up convergence.
Learning rate is set at 0.001 to balance convergence and avoiding overshooting.
Reproducibility:

Setting seeds ensures reproducibility, crucial for consistent results in machine learning experiments.
4. How to run?
The Final Project Folder includes the following files:

resnet_mixup.py: The training module
mixup_generator.py: Mixup data augmentation file
random_eraser.py: Random_eraser data augmentation
inference.py: The evaluation module
model.h5: The trained model.
Run the inference.py file to generate the submission.csv file.

5. Conclusion:
While several models were considered for transfer learning on a complex multi-dataset, ResNet50 emerged as the most successful option. The chosen model demonstrated a balance between computational efficiency and classification accuracy. The thorough evaluation of other models provided insights into their limitations, guiding the selection process based on the dataset's characteristics and available resources.


The data is linked here: https://www.kaggle.com/competitions/ucsc-cse-244-fall-2023-final-project/data
