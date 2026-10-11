# CsiONet Code

This repository provides the implementation of CsiONet presented in the following paper:

**K. Yang, Q. Xiao, C. Wang, and Z. Wang, "Deep Learning-Based Channel Prediction With Outdated CSI for Underwater Acoustic Communications," IEEE Wireless Communications Letters, vol. 15, pp. 5001–5005, 2026.**

[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11668842)

## Data Description

CsiONet predicts the current channel state information (CSI) from outdated CSI:

$$
\mathbf{h}_{t-2D-1} \xrightarrow{} \mathrm{CsiONet} \xrightarrow{} \hat{\mathbf{h}}_{t}
$$

The model input, prediction target, and model output are summarized below:

| Item | Description |
|---|---|
| Model input | Outdated CSI $\mathbf{h}_{t-2D-1}$ |
| Prediction target | Current CSI $\mathbf{h}_{t}$ |
| Model output | Predicted CSI $\hat{\mathbf{h}}_{t}$ |

The forward signal propagation requires $D$ time slots, while CSI feedback requires another $D$ time slots. Therefore, the round-trip feedback delay is $2D$ time slots. After receiving the feedback, the transmitter further predicts the channel state for the next time slot. As a result, the input CSI and the prediction target are separated by a total of $2D+1$ CSI snapshots.

In this experiment, $D=4$, and each time slot has a duration of 128 ms. Therefore:

- The round-trip feedback delay is $2D=8$ time slots, corresponding to $8 \times 128\ \mathrm{ms}=1.024\ \mathrm{s}$.
- The index difference between the input and target snapshots is $2D+1=9$, corresponding to $9 \times 128\ \mathrm{ms}=1.152\ \mathrm{s}$.
- There are 8 CSI snapshots between the input snapshot and the target snapshot.

Each processed data sample has a shape of `(4, 120)`. The four channels are defined as follows:

| Channel | Description |
|---|---|
| 0 | Real part of the outdated CSI $\mathbf{h}_{t-2D-1}$ |
| 1 | Imaginary part of the outdated CSI $\mathbf{h}_{t-2D-1}$ |
| 2 | Real part of the current CSI $\mathbf{h}_{t}$ |
| 3 | Imaginary part of the current CSI $\mathbf{h}_{t}$ |

The length of 120 represents the 120 delay taps of the channel impulse response (CIR).
