# Color-Vision Background

This page explains the color-vision science behind ColorCast's simulation, error-map, and Daltonization modules. For API details, see the module docstrings and the API reference.

---

## Human Color Vision

The human retina contains three classes of cone photoreceptors:

- **L cones** (long-wavelength / "red" sensitive, ~560 nm peak)
- **M cones** (medium-wavelength / "green" sensitive, ~530 nm peak)
- **S cones** (short-wavelength / "blue" sensitive, ~420 nm peak)

A _trichromat_ encodes incoming light as a three-dimensional signal. A _dichromat_ can only represent a two-dimensional projection of that signal because one cone class is absent or non-functional (Smith & Pokorny, 1975).

| Condition    | Missing cone | Common name |
| ------------ | ------------ | ----------- |
| Protanopia   | L            | Red-blind   |
| Deuteranopia | M            | Green-blind |
| Tritanopia   | S            | Blue-blind  |

In information-theoretic terms, dichromacy is a lossy dimensionality reduction of the color signal from three cone dimensions to two. Two colors that are perceptually distinct for a trichromat may map to the same point in the dichromat's reduced color space, making them indistinguishable. This is the core accessibility problem Daltonization addresses.

---

## Simulation Pipeline

`ColorBlindSimulator` in `colorcast/processing/simulation.py` converts an RGB image to approximate how a dichromat would perceive it. The pipeline is conceptually the standard sRGB → linear RGB → LMS → projected LMS → linear RGB → sRGB route (Brettel, Viénot, & Mollon, 1997; Viénot, Brettel, & Mollon, 1999), but the RGB → LMS → projection → LMS → RGB chain is pre-computed into a single linear RGB matrix per deficiency using the Smith-Pokorny (1975) cone fundamentals (DaltonLens, 2021b):

1. Normalize input to float32 in [0, 1]. Input is assumed to be nonlinear sRGB (CIE, 2004).
2. Gamma-decode sRGB to linear RGB.
3. Flatten (H, W, 3) to (N, 3) for vectorized matrix operations.
4. Apply the deficiency-specific 3×3 matrix in linear RGB space.
5. Gamma-encode linear RGB to nonlinear sRGB.
6. Clip to [0, 1], reshape to (H, W, 3), and return a float32 array.

Each matrix row sums to 1, so achromatic whites and grays are preserved by construction. Conceptually, the matrix still collapses 3-D LMS space onto a 2-D confusion plane: any two colors that differed only along the eliminated cone axis are mapped to a single point and become indistinguishable to the simulated observer.

### Deficiency-specific notes

- **Protanopia and Deuteranopia**: the linear RGB matrices follow the single-matrix method described by Viénot, Brettel, & Mollon (1999).
- **Tritanopia**: the implementation uses the two-half-plane construction described by Brettel, Viénot, & Mollon (1997). A pixel selects one of two pre-computed linear RGB projection matrices depending on which side of the neutral diagonal it falls in linear RGB space; the matrices already include the full RGB → LMS → projection → LMS → RGB chain.

### Why Tritanopia Looks Different

Tritanopia (S-cone loss) produces a visually distinct simulation compared with Protanopia or Deuteranopia because of the luminance contribution of each cone type.

Photopic luminance is dominated by L and M cones, while the S cone contributes a negligible amount (MacLeod & Boynton, 1979). Vos and Walraven (1971) estimate the L:M luminance-weight ratio at roughly 2:1, meaning M contributes on the order of one-third of the combined L+M luminance signal, while S contributes essentially nothing. Because of this:

- **Protanopia** removes L, which carries a large share of luminance, so the image loses contrast and saturated reds appear very dark.
- **Deuteranopia** removes M; greens shift, but perceived brightness is relatively preserved because L still carries most luminance.
- **Tritanopia** removes S, which contributes almost nothing to brightness, so overall luminance and contrast are almost unaffected. The perceptual shift is purely chromatic: short-wavelength blues and yellows become confused, while reds, greens, and overall lightness remain nearly intact.

---

## Error Maps

Phase 1 (simulation) produces an image showing what a dichromat sees. Phase 2 (`colorcast/analysis/error_map.py`) measures what was lost between the original and the simulation.

The error map provides two complementary views:

1. **Signed difference** (`original − simulated`, shape H×W×3): keeps direction. Positive values mean the original was brighter in that channel; negative values mean the simulation boosted it. This signed map is the direct input to Daltonization:

   ```python
   corrected = np.clip(original + alpha * signed, 0, 1)
   ```

2. **Absolute magnitude map** (`|original − simulated|`, same shape): unsigned “heat” showing how much color information was destroyed per pixel, regardless of direction. This is the basis of the visual heatmap and scalar summary statistics.

### Luminance Masking

A plain RGB difference conflates two independent effects:

- **Luminance** changes – shifts in perceived lightness (`L*` in CIE `Lab*`; CIE, 2004). These are mostly irrelevant for accessibility analysis because a dichromat's brightness perception is usually intact.
- **Chromaticity** changes – shifts in hue and saturation (`a*` and `b*` in `Lab*`). This is the actual color confusion we care about.

To isolate chromaticity, the error map converts the difference image to CIE `Lab*` and zeros out the `L*` channel, leaving only `a*` (red-green axis) and `b*` (blue-yellow axis). The Euclidean magnitude `sqrt(a*² + b*²)` then represents pure chromatic error – the blueprint for Daltonization.

---

## Daltonization

Daltonization (`colorcast/analysis/daltonization.py`) is a channel re-encoding strategy. Rather than trying to restore a missing cone class, it re-routes the lost chromatic information onto a surviving channel that the affected observer can perceive (Huang, Chen, Jen, & Wang, 2009; Rasche, Geist, & Westall, 2005).

Phase 2 computes:

```python
signed_error = original - simulated
```

This is the chromatic information that went missing when the original image was projected through the dichromat's reduced color space. The signed direction tells us which way the color shifted, which is critical for re-encoding.

### Shift Matrices

A 3×3 shift matrix routes each error channel into each output channel. The matrices in this implementation are design choices, not direct copies from a single published source.

For Deuteranopia and Protanopia (red-green axis missing):

```text
ΔR = 0
ΔG = 0
ΔB = 0.7·eR + 0.1·eG
```

The Blue channel is intact for both conditions. By adding the red-green error into blue, the confusable hue pair is shifted along the blue-yellow axis – an axis the dichromat can discriminate.

For Tritanopia (blue-yellow axis missing):

```text
ΔR = 0.7·eB
ΔG = 0.7·eB
ΔB = 0
```

Here the Blue channel error is re-routed into the red-green axis, which is fully functional in tritanopes. Both R and G receive the same injection so the correction appears as a luminance modulation, which is detectable even without hue discrimination.

### Perceptual Weighting

Applying correction uniformly across the image would shift colors that are already discriminable and create an over-processed look. The Phase 2 `chroma_error` map encodes, per pixel, how much chromatic information was destroyed. It is normalized to [0, 1] and used as a spatial weight:

```python
weight = chroma_error / percentile(chroma_error, 95)
correction *= weight
```

Pixels with low chroma_error are left nearly unchanged; pixels with high chroma_error receive the strongest shift.

### Luminance Preservation

After the RGB correction, a CIE `Lab*` round-trip restores the original luminance channel (`L*`):

```python
corrected_lab[:, :, 0] = original_lab[:, :, 0]
```

This preserves global brightness and ensures the only change is in the (`a*`, `b*`) chromaticity plane. The result may look color-shifted to a standard trichromat, but the chromatic contrasts that were invisible to the affected observer are now encoded in a channel they can perceive.

---

## References

- Brettel, H., Viénot, F., & Mollon, J. D. (1997). Computerized simulation of color appearance for dichromats. _Journal of the Optical Society of America A_, 14(10), 2647-2655. DOI: 10.1364/josaa.14.002647.
- Commission Internationale de l'Éclairage. (2004). _Colorimetry_ (3rd ed.) (CIE Publication No. 15:2004). CIE.
- DaltonLens. (2021a). Understanding LMS-based color blindness simulations. https://daltonlens.org/understanding-cvd-simulation/.
- DaltonLens. (2021b). Accurate SVG filters for color blindness simulation. https://daltonlens.org/cvd-simulation-svg-filters/.
- Huang, J.-B., Chen, C.-S., Jen, T.-C., & Wang, S.-J. (2009). Image recolorization for the colorblind. In _Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_, pp. 1161-1164. DOI: 10.1109/ICASSP.2009.4959795.
- MacLeod, D. I. A., & Boynton, R. M. (1979). Chromaticity diagram showing cone excitation by stimuli of equal luminance. _Journal of the Optical Society of America_, 69(8), 1183-1186. DOI: 10.1364/josa.69.001183.
- Rasche, K., Geist, R., & Westall, J. (2005). Re-coloring Images for gamuts of lower dimension. _Computer Graphics Forum_, 24(3), 423-432. DOI: 10.1111/j.1467-8659.2005.00867.x.
- Smith, V. C., & Pokorny, J. (1975). Spectral sensitivity of the foveal cone photopigments between 400 and 500 nm. _Vision Research_, 15(2), 161-171. DOI: 10.1016/0042-6989(75)90203-5.
- Viénot, F., Brettel, H., & Mollon, J. D. (1999). Digital video colourmaps for checking the legibility of displays by dichromats. _Color Research & Application_, 24(4), 243-252. DOI: 10.1002/(SICI)1520-6378(199908)24:4<243::AID-COL5>3.0.CO;2-3.
- Vos, J. J., & Walraven, P. L. (1971). On the derivation of the foveal receptor primaries. _Vision Research_, 11(8), 799-818. DOI: 10.1016/0042-6989(71)90003-4.
