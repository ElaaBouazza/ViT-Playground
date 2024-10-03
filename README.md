## Content

**[1. Introduction](#introduction)**
  * [1.1 Transformer Architecture](#transformer-architecture)
  * [1.2. Vision Transformer](#vision-transformer)

**[2. Tasks](#tasks)**

  * [2.1. Setting up the environement](#setting-up-the-environment)
  * [2.2. Finetune Model](#finetune-model)
  * [2.3. Modifying the model](#modifying-the-model)
    * [2.3.1 Classification head](#classification-head)
    * [2.3.2 Injecting the Transformer](#injecting-the-transformer)

**[3. Results](#results)**

**[4. References](#references)**

# Introduction

## Transformer Architecture 

Transformers have accelerated the development of new techniques and models for natural language processing (NLP) tasks. While it has mostly been used for NLP tasks, it is now seeing heavy adoption in other areas such as computer vision and reinforcement learning. That makes it one of the most important modern concepts to understand and be able to apply.

If you need more information about the Transformer architecture, [this link](https://github.com/dair-ai/Transformers-Recipe) could help! 

## Vision Transformer 
This repo presents a PyTorch reimplementation of [Google's repository for the ViT model](https://github.com/google-research/vision_transformer) that was released with the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).

This paper shows that Transformers applied directly to image patches and pre-trained on large datasets work really well on image recognition tasks.


<p align="center" width="100%">
    <img width="70%" src="./img/figure1.png">
</p>

Vision Transformer achieves State-of-the-Art image recognition tasks with a standard Transformer encoder and fixed-size patches. In order to perform classification, the author uses the traditional approach of adding an extra learnable "classification token" to the sequence.


# Tasks
## Setting up the environment
Most of the used packages are included in the `requirements.txt`. You should be able to setup the environment locally (with GPU support) or use google colab.

> <span style="color:red">**[Task 1] You need to provide the correct versions (in requirements.txt) for each library used in your project.**</span>

## Download the Pre-trained model (Google's Official Checkpoint)

```
# imagenet21k pre-train
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
```

The checkpoint should be stored under `./checkpoints/`, for instance, to launch the pretraining. 


> <span style="color:red">**[Task 2] Use [hydra](https://hydra.cc/docs/intro/) to pass arguments to the script instead of hardcoding them.**</span>

## Finetune Model

Dataset: CIFAR-10 will be automatically downloaded and used for finetuning. 

[Pytorch Lightning](https://lightning.ai/) is used to train/finetune the model. You can change the parameters of the `Trainer` and the `batch_size` to match your hardware specifications. 

Command:
```
python main.py 
```

> <span style="color:red">**[Task 3] The learning rate (lr) is a very sensible hyperparameter. Include an lr-scheduler in the trainer.**</span>

> <span style="color:red">**[Task 4] Finetune the pre-trained model and put the results in [this section](#results) .**</span>

## Modifying the model
### **Classification head**
The previous task performs a finetuning that includes all model's weights. Therefore, we want to explore finetuning only the classification head instead of changing the transformer's pre-trained weights.

<p align="center" width="100%">
    <img width="50%" src="./img/figure2.jpg">
</p>


> <span style="color:red">**[Task 5] Propose a new classification head, [freeze](https://jimmy-shen.medium.com/pytorch-freeze-part-of-the-layers-4554105e03a6) the transformer encoder then finetune the model. Put the results in [this section](#results) .**</span>


### **Injecting the Transformer**

In the previous tasks, we performed gradient-based finetuning on a downstream task (classification). While fine-tuning a pre-trained model has produced many state-of-the-art results, it makes the model specialized for a single task with an *'entirely'* new set of parameter values. This can be impractical when finetuning a model on many downstream tasks.

In this task, we will add modifications to the transformer encoder that could allow efficient and cheaper finetuning (in terms of computations).

We will introduce trainable vectors  $l_W$ into different components of the transformer encoder, which perform element-wise rescaling of inner model activations. For any model layer expressed as a matrix multiplication of the form $h=Wx$
, it, therefore, performs an element-wise multiplication with $l_W$
, such that:

$$ h=l_W⊙Wx $$

where $⊙$ denotes element-wise multiplication: the entries of $l_W$
 are broadcasted to the shape of $W$
.
<p align="center" width="100%">
    <img width="50%" src="./img/inject.jpg">

    Illustration of injection method within one transformer layer. Trained components are colored in shades of magenta. 
</p>



> <span style="color:red">**[Task 6] Implement the *injection* approach in the transformer encoder architecture.**</span>

> <span style="color:red">**[Task 7] Adjust the optimizer in order to make only the added vectors trainable within the transformer architecture. PS: The Classification head (MLP or the one created in task 5) should be trainable too.**</span>

<span style="color:green"> 

**[Get Some Bonuses 🤑 ]** The implementation should allow: 
- Easy changing the injection architecture [*(Hint)*](https://refactoring.guru/design-patterns)
-  Control the injection (off/on) by passing a hydra argument.
- Storing only the trainable parameters (injection's parameters + classification head)

**Anything that can make the code more clear/clean/modular/.. is a plus.**

</span>

</p>


# Results

**You should update this file to submit your results.**

<p align="center" width="100%">

|    model       |  acc   |
|:--------------:|:------:|
|  Full model    |  0.97  |
| Classification head  |  ----  |
| Injected model |  ----  |

</p>

# References
* [Google ViT](https://github.com/google-research/vision_transformer)


