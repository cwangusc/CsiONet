"""Evaluate CsiONet on the final record at multiple SNR levels.

The processed data has shape (N, 4, 120). Its four channels contain the real
and imaginary parts of the outdated CSI, followed by the real and imaginary
parts of the current CSI. The input file contains the final record only, and
its final approximately 6.3-second segment forms the test set. See README.md
for details about data generation, delay relationships, and dataset splits.
"""

import argparse
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "model" / "CsiONet.pt"

VAL_END = 192
TEST_END = 241
TEST_SIZE = TEST_END - VAL_END
SLOT_DURATION_SECONDS = 0.128
TEST_DURATION_SECONDS = TEST_SIZE * SLOT_DURATION_SECONDS
SNR_LEVELS: tuple[float | None, ...] = (None, 10.0, 5.0, 0.0, -5.0)


def load_test_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    array = np.load(path, allow_pickle=False)
    expected_shape = (TEST_END, 4, 120)

    if array.shape != expected_shape:
        raise ValueError(
            f"The input must contain the complete processed dataset with shape "
            f"{expected_shape}; received {array.shape}."
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"The input must be a numeric array; received {array.dtype}.")
    
    # Use only the final approximately 6.3-second test segment of the record.
    test_data = array[VAL_END:TEST_END]
    test_input = np.ascontiguousarray(test_data[:, 0:2, :], dtype=np.float32)
    test_target = np.ascontiguousarray(test_data[:, 2:4, :], dtype=np.float32)
    return test_input, test_target


def add_awgn(
    input_array: np.ndarray,
    snr_db: float | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float | None]:
    if snr_db is None:
        return input_array.copy(), None

    sample_count, _, sequence_length = input_array.shape
    signal_power = np.sum(np.square(input_array, dtype=np.float64)) / (
        sample_count * sequence_length
    )
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(input_array.shape).astype(np.float32)
    noise *= np.float32(np.sqrt(noise_power / 2.0))

    measured_noise_power = np.sum(np.square(noise, dtype=np.float64)) / (
        sample_count * sequence_length
    )
    measured_snr = 10.0 * np.log10(signal_power / measured_noise_power)

    noisy_input = np.ascontiguousarray(input_array + noise, dtype=np.float32)
    return noisy_input, float(measured_snr)


def predict(
    model: torch.jit.ScriptModule,
    input_array: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    results: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, input_array.shape[0], batch_size):
            batch = torch.from_numpy(input_array[start:start + batch_size]).to(device)
            output = model(batch)
            results.append(output.detach().cpu().numpy())

    return np.concatenate(results, axis=0)


def calculate_nmse(target: np.ndarray, prediction: np.ndarray) -> float:
    """Match the sample-wise complex NMSE calculation used by the source project."""
    if target.shape != prediction.shape:
        raise ValueError(
            f"Target and prediction shapes must match; received "
            f"{target.shape} and {prediction.shape}."
        )
    if target.ndim != 3 or target.shape[1] != 2:
        raise ValueError(
            f"NMSE expects arrays with shape (N, 2, L); received {target.shape}."
        )

    target_complex = target[:, 0, :] + 1j * target[:, 1, :]
    prediction_complex = prediction[:, 0, :] + 1j * prediction[:, 1, :]

    error_energy = np.sum(np.abs(target_complex - prediction_complex) ** 2, axis=1)
    target_energy = np.sum(np.abs(target_complex) ** 2, axis=1)
    if np.any(target_energy == 0):
        raise ZeroDivisionError("NMSE is undefined for a zero-energy target sample.")

    return float(np.mean(error_energy / target_energy))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CsiONet NMSE on the final record at multiple SNR levels."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_DIR / "data" / "NOF1_Last_Record.npy",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2023,
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA-capable GPU is available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    test_input, test_target = load_test_data(args.input)
    model = torch.jit.load(str(args.model), map_location=device)
    model.eval()

    print(
        "CsiONet NMSE on the final record "
        f"(approximately {TEST_DURATION_SECONDS:.1f} s of test data)"
    )
    for offset, snr_db in enumerate(SNR_LEVELS):
        noisy_input, measured_snr = add_awgn(
            test_input,
            snr_db,
            np.random.default_rng(args.seed + offset),
        )
        prediction = predict(model, noisy_input, device, args.batch_size)
        nmse = calculate_nmse(test_target, prediction)
        nmse_db = 10.0 * np.log10(
            max(nmse, np.finfo(np.float64).tiny)
        )

        if snr_db is None:
            print(f"SNR = inf dB: NMSE = {nmse_db:.4f} dB")
        else:
            print(
                f"SNR = {snr_db:g} dB: "
                f"NMSE = {nmse_db:.4f} dB"
            )


if __name__ == "__main__":
    main()
