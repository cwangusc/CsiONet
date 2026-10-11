# CsiONet

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python CsiONet.py
```

The program evaluates CsiONet on the final record and prints the NMSE in dB at
infinite, 10 dB, 5 dB, 0 dB, and -5 dB input SNR conditions.


## Data Description

CsiONet predicts the current channel state from outdated channel state
information (CSI):

\[
\mathbf{h}_{t-2D-1}
\;\xrightarrow{\mathrm{CsiONet}}\;
\hat{\mathbf{h}}_{t}
\]

The model input is the outdated CSI \(\mathbf{h}_{t-2D-1}\), the prediction
target is the current CSI \(\mathbf{h}_{t}\), and the model output is the
prediction \(\hat{\mathbf{h}}_{t}\).

Forward propagation takes \(D\) time slots, and CSI feedback takes another
\(D\) time slots, resulting in a round-trip feedback delay of \(2D\) time
slots. After receiving the feedback, the transmitter predicts the channel in
the next time slot. The input and prediction target are therefore separated by
\(2D+1\) CSI snapshots.

In this experiment, \(D=4\), and each time slot is 128 ms. Therefore:

- The round-trip feedback delay is \(2D=8\) time slots, or
  \(8 \times 128\ \mathrm{ms}=1.024\ \mathrm{s}\).
- The input and target snapshot indices differ by \(2D+1=9\), corresponding to
  \(9 \times 128\ \mathrm{ms}=1.152\ \mathrm{s}\).
- Eight intermediate CSI snapshots lie between the input and target snapshots.

Each processed sample has shape `(4, 120)`. The channels are:

| Channel | Description |
|---|---|
| 0 | Real part of the outdated CSI \(\mathbf{h}_{t-2D-1}\) |
| 1 | Imaginary part of the outdated CSI \(\mathbf{h}_{t-2D-1}\) |
| 2 | Real part of the current CSI \(\mathbf{h}_{t}\) |
| 3 | Imaginary part of the current CSI \(\mathbf{h}_{t}\) |

The length of 120 represents 120 delay taps in the channel impulse response
(CIR), not 120 historical time steps.

## Test Set and NMSE Evaluation

The input file `data/NOF1_Last_Record.npy` contains only the final record and
has shape `(241, 4, 120)`. Its data is stored in training, validation, and test
order. The inference program extracts the test segment with:

```python
test_data = array[VAL_END:TEST_END]
```

The current boundaries are:

```python
VAL_END = 192
TEST_END = 241
```

The test set is therefore `array[192:241]`. With a time-slot duration of
128 ms, this segment represents approximately 6.3 seconds of test data.

At each input SNR, the program reconstructs the complex-valued target and
prediction from their real and imaginary channels. It first calculates the
sample-wise linear NMSE:

\[
\mathrm{NMSE}
=
\frac{1}{N}
\sum_{i=1}^{N}
\frac{\lVert\mathbf{h}_i-\hat{\mathbf{h}}_i\rVert_2^2}
     {\lVert\mathbf{h}_i\rVert_2^2}.
\]

The reported value is converted to decibels using:

\[
\mathrm{NMSE}_{\mathrm{dB}}=10\log_{10}(\mathrm{NMSE}).
\]
