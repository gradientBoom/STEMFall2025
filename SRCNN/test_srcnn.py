import torch
import numpy as np
import pickle
from srcnn_model import SRCNN
from L1_minimization_recovering_code.main import random_destroy_to_image
from L1_minimization_recovering_code.L1_minimization import do_Fourier_transform
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


def load_cifar10_test_batch(batch_file):
    with open(batch_file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']
        images = data.reshape(-1, 3, 32, 32).astype('float32') / 255.0
        images_gray = np.mean(images, axis=1, keepdims=True)
        return images_gray  # Shape: (N,1,32,32)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SRCNN().to(device)
    model.load_state_dict(torch.load("srcnn_cifar_inpainting.pth"))
    model.eval()

    test_images = load_cifar10_test_batch("cifar-10-batches-py/test_batch")

    total_psnr, total_ssim = 0, 0

    for idx in range(10):  # Evaluate on first 10 images
        img = test_images[idx].squeeze()  # (32,32)

        # Use project Masking function
        masked_img, _ = random_destroy_to_image(img * 255, missing_value=0, percentage=0.3)
        masked_img = masked_img / 255.0

        input_tensor = torch.tensor(masked_img).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor).cpu().squeeze().numpy()

        img_gt = img
        img_masked = masked_img
        img_recovered = np.clip(output, 0, 1)

        psnr_val = psnr(img_gt, img_recovered)
        ssim_val = ssim(img_gt, img_recovered)

        total_psnr += psnr_val
        total_ssim += ssim_val

        # Fourier Spectrum Analysis
        fft_orig = np.abs(do_Fourier_transform(img_gt * 255))
        fft_masked = np.abs(do_Fourier_transform(img_masked * 255))
        fft_recovered = np.abs(do_Fourier_transform(img_recovered * 255))

        plt.figure(figsize=(12,4))
        plt.subplot(2,4,1); plt.title("Original"); plt.imshow(img_gt, cmap='gray')
        plt.subplot(2,4,2); plt.title("Masked"); plt.imshow(img_masked, cmap='gray')
        plt.subplot(2,4,3); plt.title("Recovered"); plt.imshow(img_recovered, cmap='gray')
        plt.subplot(2,4,4); plt.axis('off'); plt.text(0.1,0.5,f"PSNR: {psnr_val:.2f}\nSSIM: {ssim_val:.4f}")

        plt.subplot(2,4,5); plt.title("FFT Original"); plt.imshow(np.log1p(fft_orig), cmap='gray')
        plt.subplot(2,4,6); plt.title("FFT Masked"); plt.imshow(np.log1p(fft_masked), cmap='gray')
        plt.subplot(2,4,7); plt.title("FFT Recovered"); plt.imshow(np.log1p(fft_recovered), cmap='gray')
        plt.tight_layout(); plt.show()

    print(f"Average PSNR: {total_psnr/10:.2f} dB")
    print(f"Average SSIM: {total_ssim/10:.4f}")