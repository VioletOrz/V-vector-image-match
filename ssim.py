from Violet.ImageMatch.moudels.SSIM import calculate_ssim, calculate_ssim_binarize

print(calculate_ssim('1.png', '2.png'))
print(calculate_ssim('1.png', '3.png'))
print(calculate_ssim('2.png', '3.png'))