# Why Zero-DCE++ outputs can look too blue

The most likely cause is the zero-reference color-constancy objective, not a BGR/RGB file-loading bug.

## Primary cause: gray-world color constancy

`L_color` computes the global mean of the enhanced output in the red, green, and blue channels, then penalizes the distance between those three means. This is a gray-world assumption: the average scene color should be neutral gray. If the original image is warm, sunset-lit, tungsten-lit, or naturally red/yellow, forcing the output channel averages to match can raise the blue channel or suppress red/green, creating a colder blue cast.

This effect is amplified during training because `lowlight_train.py` gives the color loss a fixed weight and optimizes it with the exposure, spatial-consistency, and total-variation losses. There is no paired ground-truth color target or white-balance target that tells the model to preserve the original scene chromaticity.

## Secondary cause: fixed exposure target

`L_exp` pushes the average patch brightness toward a fixed scalar target. In `lowlight_train.py`, that target is set to `E = 0.6`. When dark, warm images are brightened to this global target, the color-constancy loss can use the blue channel as an easy way to neutralize the brightened result.

## Less likely: RGB/BGR channel swap

The current data path uses PIL and NumPy, then permutes from HWC to CHW. PIL loads normal RGB images as RGB, and `torchvision.utils.save_image` writes tensors in RGB order. A true BGR/RGB swap would usually make reds and blues exchange dramatically, not merely make outputs cooler. Still, if custom OpenCV code is added later, check whether `cv2.imread` BGR images are converted with `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` before passing them into the network.

## How to reduce the blue cast

1. Lower the color-loss weight, for example change `loss_col = 5 * ...` to `loss_col = 1 * ...` or expose it as a command-line option.
2. Replace gray-world color constancy with a color-preservation term that compares input and output chromaticity, such as normalized RGB or Lab `a/b` consistency.
3. Use paired or pseudo-paired color targets when training for a specific camera/domain.
4. Add a white-balance calibration step before enhancement if the camera has a known color temperature bias.
5. Reduce the fixed exposure target for scenes where warm highlights or dusk/sunset lighting should be preserved.
6. Validate channel order with a small red/green/blue test image before blaming the model.

## Minimal code change to try first

The training script exposes configurable loss weights, so run ablations such as:

```text
python lowlight_train.py --color_loss_weight 1.0 --exposure_target 0.45
```

If the blue cast drops when the color-loss weight drops, the color-constancy term is the cause. If red and blue are fully swapped, inspect the image I/O path instead.
