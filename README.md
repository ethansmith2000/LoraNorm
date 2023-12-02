# LoraNorm

LoraNorm is another example of PEFT (Parameter Efficient FineTuning) that targets the affine weights of norm layers
While a lot of different LoRA schemes have been used with Stable Diffusion, I have not seen any that make the norms
trainable.

Disclaimer: Technically this is not a LoRA as we're not training matrices but single vectors, but I figured the name 
gets the idea across, and  it can be loaded on top of any Stable Diffusion model in the same manner.

The collective weights and biases of the norm layers of Stable Diffusion 1.5 only comes out to 200k parameters.
or 0.02% of the model. I originally set this up to augment current LoRA training but it seems training even in isolation
does a decent job!

I hope this opens up doors to exploring more the roles of affine weights in norm layers in text to image generation :D

## Usage

## Installation
```commandline
pip install -r requirements.txt
```

## Training
```commandline
bash train.sh
```

Then once trained, you can run inference with the provided notebook
