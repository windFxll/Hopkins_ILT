from ilt.optimization.losses import total_variation


def haar_details(mask, levels=3):
    """Mean absolute Haar detail coefficients, normalized by scale."""
    if any(size % (2 ** levels) for size in mask.shape):
        raise ValueError("Mask dimensions must be divisible by 2 ** Haar levels")
    result = mask.new_zeros(())
    low = mask
    for level in range(levels):
        a, b = low[0::2, 0::2], low[0::2, 1::2]
        c, d = low[1::2, 0::2], low[1::2, 1::2]
        bands = ((a + b - c - d), (a - b + c - d), (a - b - c + d))
        result = result + sum(band.abs().mean() for band in bands) / (2 ** (level + 1))
        low = (a + b + c + d) * 0.5
    return result


def squared_laplacian(mask):
    """Five-point Laplacian energy with the optical model's periodic boundary."""
    laplacian = (mask.roll(1, 0) + mask.roll(-1, 0) + mask.roll(1, 1)
                 + mask.roll(-1, 1) - 4.0 * mask)
    return laplacian.square().mean()


SPATIAL_PENALTIES = {
    "tv": total_variation,
    "wavelet": haar_details,
    "laplacian": squared_laplacian,
}
