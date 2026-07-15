# Upgrade plan: denoising, defogging, and sun-flare brightness reduction

Zero-DCE++ currently estimates an illumination curve with a lightweight CNN and trains it with zero-reference losses for exposure, color constancy, spatial consistency, and illumination smoothness. That makes it a strong low-light enhancer, but it does not explicitly model sensor noise, atmospheric haze/fog, or localized saturated glare. The safest upgrade is to keep the curve-estimation backbone as the illumination branch and add task-specific heads plus losses that can be trained with either paired data, synthetic degradation, or zero-reference priors.

## 1. Replace the single-output enhancer with a multi-task restoration model

Add a shared encoder and four lightweight heads:

1. **Illumination curve head**: keep the current Zero-DCE++ curve output for exposure correction.
2. **Denoising head**: predict either a residual noise map or a clean RGB image. A residual head is usually easier to add: `clean = enhanced - noise_residual`.
3. **Defog / dehaze head**: predict transmission `t(x)` and atmospheric light `A`, then recover radiance with `J(x) = (I(x) - A) / max(t(x), t0) + A`.
4. **Flare suppression head**: predict a soft flare mask and a highlight attenuation map so only saturated, low-detail flare regions are darkened.

Recommended output dictionary:

```python
{
    "enhanced": enhanced_image,
    "curve": curve_map,
    "denoised": denoised_image,
    "transmission": transmission_map,
    "atmospheric_light": atmospheric_light,
    "flare_mask": flare_mask,
    "flare_reduced": final_image,
}
```

Keep these heads shallow if mobile speed matters. For higher quality, replace the shared encoder with a small U-Net, Restormer-like block stack, or NAFNet-style blocks.

## 2. Add losses for each degradation

The existing training loss should stay as the baseline low-light constraint, but each new task needs a direct signal.

### Denoising losses

Use one or more of:

- `L1` or Charbonnier loss against paired clean targets.
- Noise-to-noise or blind-spot loss when clean targets are unavailable.
- Edge-preserving loss so denoising does not smear small structure.
- Frequency-domain loss if high-ISO chroma noise remains visible.

### Defog / dehaze losses

Use one or more of:

- Reconstruction loss on synthetic hazy/clean pairs generated with the atmospheric scattering model.
- Transmission smoothness loss weighted by image edges.
- Dark-channel or contrast prior as a zero-reference regularizer.
- Color-consistency loss to avoid over-blue or over-yellow outputs.

### Sun-flare / glare losses

Use one or more of:

- Flare mask supervision from synthetic flare overlays or manually labeled masks.
- Highlight compression loss that penalizes only high-luminance saturated regions.
- Local contrast preservation loss outside the flare mask.
- Identity loss on non-flare regions so the model does not globally darken the whole image.

A useful flare-region target is:

```python
luma = 0.299 * r + 0.587 * g + 0.114 * b
candidate_mask = sigmoid((luma - 0.85) * 20) * low_texture_mask
```

Then train the attenuation head to reduce `candidate_mask` regions while preserving everything else.

## 3. Expand the training data

Zero-reference losses are not enough for all three new degradations. Use a mixed training set:

- **Real low-light images**: keep the existing training set for exposure correction.
- **Noisy images**: add real high-ISO pairs such as SIDD/DND, or synthesize Poisson-Gaussian noise on clean images.
- **Foggy images**: use RESIDE-style synthetic/real hazy data, plus your target-domain fog images.
- **Sun-flare images**: synthesize flare streaks, veiling glare, and saturated disks using alpha-blended flare layers; add real dashcam/drone/outdoor examples if available.

Batch mixing should sample one restoration objective per batch or use task flags so every mini-batch does not need every annotation type.

## 4. Training schedule

1. **Warm start from Zero-DCE++**: load the existing checkpoint into the illumination branch.
2. **Train heads independently**: freeze the shared encoder for a few epochs and train denoise, defog, and flare heads on their data.
3. **Joint fine-tune**: unfreeze everything with a small learning rate and balanced loss weights.
4. **Target-domain fine-tune**: fine-tune on the exact camera/domain where fog, noise, and sun flares occur.

Loss balancing should begin conservatively:

```text
loss = zero_dce_loss
     + 1.0 * denoise_loss
     + 0.5 * dehaze_loss
     + 0.5 * flare_loss
     + 0.1 * identity_loss_outside_masks
```

Tune by visual inspection and validation metrics rather than assuming fixed weights will transfer.

## 5. Inference pipeline option

If retraining a single multi-task model is too large a change, add a staged inference mode first:

1. Run Zero-DCE++ illumination enhancement.
2. Run a denoiser on the enhanced image.
3. Run dehazing/defogging only when a fog detector or low-contrast metric triggers it.
4. Run flare attenuation only on saturated, low-texture highlight masks.

This is slower than one model but easier to debug because each stage can be evaluated independently.

## 6. Validation metrics

Use both full-reference and no-reference checks:

- Denoising: PSNR, SSIM, LPIPS, and crop-level noise residual inspection.
- Defogging: PSNR/SSIM on synthetic haze, FADE or contrast metrics on real fog, plus color-shift checks.
- Flare reduction: masked highlight intensity reduction, non-mask identity error, and local contrast around the sun.
- Overall: NIQE/BRISQUE, user preference tests, and failure-case galleries.

## 7. Implementation checklist

- Add a `RestorationNet` model beside the current `enhance_net_nopool` instead of replacing it immediately.
- Refactor training so losses are configured by command-line flags.
- Add dataset loaders that can return optional fields: `clean`, `noise_level`, `fog_transmission`, and `flare_mask`.
- Save intermediate outputs during validation: enhanced, denoised, defogged, flare mask, and final image.
- Keep the existing Zero-DCE++ scripts working for backward compatibility.

## Recommended first milestone

The first milestone is implemented in code as a lightweight flare suppression head:

1. `enhance_net_nopool` now predicts a `flare_mask` from decoder features.
2. The model returns both the standard `enhanced_image` and `flare_reduced_image`.
3. `L_flare` builds a soft high-luminance candidate mask, trains the predicted mask toward it, compresses bright flare candidates, and preserves non-flare regions.
4. `lowlight_test.py` writes the flare-reduced output image.

Next, add denoising and then defogging as separate heads or staged modules.
