import os
import cv2 as cv
import numpy as np
import torch
from srcnn_model import SRCNN
from L1_minimization_recovering_code.main import random_destroy_to_image, get_report_name
from L1_minimization_recovering_code.L1_minimization import do_Fourier_transform, do_image_recovery
from L1_minimization_recovering_code.util import get_test_image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Load test data
    test_images, file_names = get_test_image("cifar-10-batches-py/test_batch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load SRCNN model
    model = SRCNN().to(device)
    model.load_state_dict(torch.load("srcnn_cifar_inpainting.pth"))
    model.eval()

    save_path = "reports_l1_vs_srcnn"
    os.makedirs(save_path, exist_ok=True)

    log_path = os.path.join(save_path, "evaluation_l1_vs_srcnn.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    total_l1_psnr, total_l1_ssim, total_l1_err = 0, 0, 0
    total_srcnn_psnr, total_srcnn_ssim, total_srcnn_err = 0, 0, 0

    for idx in range(5):
        # Convert to grayscale
        img_color = test_images[idx]
        img_gray = cv.cvtColor(img_color, cv.COLOR_RGB2GRAY).astype(np.float32)

        # Generate mask and missing coordinates
        masked_img, missing_coords = random_destroy_to_image(img_gray, missing_value=0, percentage=0.3)
        masked_coords_set = set(map(tuple, missing_coords))

        # ---- SRCNN Restoration ----
        input_tensor = torch.tensor(masked_img / 255.0).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            srcnn_out = model(input_tensor).cpu().squeeze().numpy()
        srcnn_restored = np.clip(srcnn_out * 255.0, 0, 255).astype(np.uint8)

        # ---- L1 Restoration ----
        l1_restored, _ = do_image_recovery(masked_img, masked_coords_set, threshold=0.2)
        l1_restored = np.clip(l1_restored, 0, 255).astype(np.uint8)

        # ---- Metrics ----
        psnr_l1 = psnr(img_gray, l1_restored, data_range=255)
        ssim_l1 = ssim(img_gray, l1_restored, data_range=255)
        psnr_srcnn = psnr(img_gray, srcnn_restored, data_range=255)
        ssim_srcnn = ssim(img_gray, srcnn_restored, data_range=255)

        # ---- Binary Error (Mismatch) ----
        binary_gt = (img_gray >= 127).astype(int)
        binary_l1 = (l1_restored >= 127).astype(int)
        binary_srcnn = (srcnn_restored >= 127).astype(int)
        masked = (masked_img == 0)

        mismatch_l1 = np.sum(binary_gt[masked] != binary_l1[masked])
        mismatch_srcnn = np.sum(binary_gt[masked] != binary_srcnn[masked])
        total_masked = np.sum(masked)

        err_l1 = mismatch_l1 / total_masked
        err_srcnn = mismatch_srcnn / total_masked

        # Log to txt
        log_file.write(f"Image {idx + 1}: {file_names[idx]}\n")
        log_file.write(f"L1     - Masked: {total_masked}, Mismatches: {mismatch_l1}, Error rate: {err_l1:.2%}\n")
        log_file.write(
            f"SRCNN  - Masked: {total_masked}, Mismatches: {mismatch_srcnn}, Error rate: {err_srcnn:.2%}\n\n")

        # Accumulate totals
        total_l1_psnr += psnr_l1
        total_l1_ssim += ssim_l1
        total_l1_err += err_l1

        total_srcnn_psnr += psnr_srcnn
        total_srcnn_ssim += ssim_srcnn
        total_srcnn_err += err_srcnn

        # ---- FFT ----
        fft_orig = np.abs(do_Fourier_transform(img_gray))
        fft_masked = np.abs(do_Fourier_transform(masked_img))
        fft_l1 = np.abs(do_Fourier_transform(l1_restored))
        fft_srcnn = np.abs(do_Fourier_transform(srcnn_restored))

        # ---- Plot ----
        plt.figure(figsize=(14, 6))

        plt.subplot(2, 5, 1); plt.title("Original"); plt.imshow(img_gray, cmap='gray')
        plt.subplot(2, 5, 2); plt.title("Masked"); plt.imshow(masked_img, cmap='gray')
        plt.subplot(2, 5, 3); plt.title("L1 Recovered"); plt.imshow(l1_restored, cmap='gray')
        plt.subplot(2, 5, 4); plt.title("SRCNN Recovered"); plt.imshow(srcnn_restored, cmap='gray')
        plt.subplot(2, 5, 5); plt.axis('off');
        plt.text(0.1, 0.5, f"L1 PSNR: {psnr_l1:.2f}\nL1 SSIM: {ssim_l1:.4f}\n\nSRCNN PSNR: {psnr_srcnn:.2f}\nSRCNN SSIM: {ssim_srcnn:.4f}", fontsize=10)

        plt.subplot(2, 5, 6); plt.title("FFT Original"); plt.imshow(np.log1p(fft_orig), cmap='gray')
        plt.subplot(2, 5, 7); plt.title("FFT Masked"); plt.imshow(np.log1p(fft_masked), cmap='gray')
        plt.subplot(2, 5, 8); plt.title("FFT L1"); plt.imshow(np.log1p(fft_l1), cmap='gray')
        plt.subplot(2, 5, 9); plt.title("FFT SRCNN"); plt.imshow(np.log1p(fft_srcnn), cmap='gray')

        plt.tight_layout()

        # Save report
        report_name = get_report_name(file_names[idx])
        report_name = report_name.replace(".png", "_compare.png")
        plt.savefig(os.path.join(save_path, report_name), bbox_inches='tight')
        plt.close()

        print(f"Saved comparison report for image {idx + 1}: {report_name}")

    log_file.write("[L1     EVALUATION] Avg PSNR: {:.2f} dB, Avg SSIM: {:.4f}, Avg Error Rate: {:.2%}\n".format(
        total_l1_psnr / 5, total_l1_ssim / 5, total_l1_err / 5))
    log_file.write("[SRCNN  EVALUATION] Avg PSNR: {:.2f} dB, Avg SSIM: {:.4f}, Avg Error Rate: {:.2%}\n".format(
        total_srcnn_psnr / 5, total_srcnn_ssim / 5, total_srcnn_err / 5))
    log_file.close()

    print(f"Saved evaluation report to: {log_path}")
