import torch
import numpy as np
import pickle
from srcnn_model import SRCNN
from L1_minimization_recovering_code.main import random_destroy_to_image, get_report_name
from L1_minimization_recovering_code.L1_minimization import do_Fourier_transform
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os
import cv2 as cv
from L1_minimization_recovering_code.util import generate_report

def load_cifar10_test_batch(batch_file):
    with open(batch_file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']
        images = data.reshape(-1, 3, 32, 32).astype('float32') / 255.0
        images_gray = np.mean(images, axis=1, keepdims=True)
        return images_gray  # Shape: (N,1,32,32)


if __name__ == "__main__":

    import sys

    sys.stdout = open(
        "reports/evaluation_report.txt", "w",
        encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SRCNN().to(device)
    model.load_state_dict(torch.load("srcnn_cifar_inpainting.pth"))
    model.eval()

    from L1_minimization_recovering_code.util import get_test_image
    test_images, file_names = get_test_image("cifar-10-batches-py/test_batch")

    total_psnr, total_ssim = 0, 0
    total_error = 0

    for idx in range(10):  # Evaluate on first 10 images
        img = cv.cvtColor(test_images[idx], cv.COLOR_RGB2GRAY).astype(np.float32) / 255.0

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
        ssim_val = ssim(img_gt, img_recovered, data_range=1.0)

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
        plt.tight_layout()


        binary_gt = (img_gt >= 0.5).astype(int)
        binary_recovered = (img_recovered >= 0.5).astype(int)
        masked_positions = (img_masked == 0)
        mismatches = np.sum(binary_gt[masked_positions] != binary_recovered[masked_positions])
        total_masked = np.sum(masked_positions)
        error_rate = mismatches / total_masked

        print(f"Masked inputs: {total_masked}")
        print(f"Mismatches: {mismatches}")
        print(f"Error rate: {error_rate:.2%}")
        total_error += error_rate

        report_name = get_report_name(file_names[idx])  # e.g. "some_img.png" -> "some_img_report.png"
        report_name = report_name.replace(".png", "_srcnn.png")  # 区分其他方法
        os.makedirs("reports",
                    exist_ok=True)
        generate_report(img_gt * 255, img_masked * 255, img_recovered * 255,
                        report_name, save_path="reports", show_report=False)

    print(f"[SRCNN EVALUATION] Average PSNR: {total_psnr / 10:.2f} dB")
    print(f"Masked inputs: {int(0.3 * 32 * 32)}")
    print(f"[SRCNN EVALUATION] Average SSIM: {total_ssim / 10:.4f}")
    print(f"[SRCNN EVALUATION] Average Error rate: {total_error / 10:.4f}")
    print(f"This script evaluates SRCNN recovery")
