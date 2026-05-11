# BCI Motor Imagery Pipeline

This repository prepares Cho2017 MOABB EEG data, trains an EEG transformer, and
optionally performs subject-specific LoRA fine-tuning.

Run the scripts in this order:

1. `moabb_data.py`
2. `moabb_train.py`
3. `lora.py`


## Environment

Install the project dependencies first:

```powershell
pip install -r requirements.txt
```

If you use the local conda environment shown in the notebooks, run scripts with:

```powershell
C:\Users\sunny\miniconda3\envs\bci\python.exe <script_name>.py
```


## 1. Prepare Data With `moabb_data.py`

`moabb_data.py` downloads Cho2017 data through MOABB, extracts EEG epochs,
selects the configured 32-channel montage, applies PyGEDAI preprocessing, and
downsamples the tensors.

Default command:

```powershell
python moabb_data.py
```

Main outputs:

```text
raw/subject_*.pt
pygedai_processed/subject_*.pt
downsampled/subject_*.pt
```

By default, channel selection uses:

```text
64ch_to_32ch_mapping.csv
```

The saved tensors are shaped like:

```text
(trials, channels, timepoints)
```

For the current 32-channel setup, downsampled files are typically:

```text
(200, 32, 385)
```

Useful options:

```powershell
python moabb_data.py --channel-mapping-csv 64ch_to_32ch_mapping.csv
python moabb_data.py --max-epochs 200
python moabb_data.py --downsample-ratio 4
python moabb_data.py --data-dir raw --downsampled-dir downsampled
```

Notes:

- Existing cached files with the wrong channel count are regenerated.
- The default downsampling ratio is 4.
- `downsampled/` is the default training input directory.


## 2. Train the Base Transformer With `moabb_train.py`

`moabb_train.py` trains the base EEG transformer on the cached downsampled data.

Default command:

```powershell
python moabb_train.py
```

Main inputs:

```text
downsampled/subject_*.pt
```

Main outputs:

```text
best_model.pt
loss.png
```

Useful options:

```powershell
python moabb_train.py --epochs 20
python moabb_train.py --batch-size 32
python moabb_train.py --lr 0.001
python moabb_train.py --num-subjects 52
python moabb_train.py --checkpoint-path best_model.pt
```

The model automatically uses the channel count from the data:

```text
n_channels = tensor.shape[2]
```

So it works with the current 32-channel tensors. A checkpoint trained on
64-channel data will not load into a 32-channel model because the spatial
convolution has a different weight shape.


## 3. Subject-Specific LoRA Fine-Tuning With `lora.py`

`lora.py` loads the base transformer checkpoint, freezes it, inserts LoRA
adapters into selected Linear layers, and fine-tunes one subject-specific model
per subject.

Default command:

```powershell
python lora.py
```

Main inputs:

```text
downsampled/subject_*.pt
best_model.pt
```

Main outputs:

```text
finetune/subject_<id>_model.pt
```

Fine-tune all subjects:

```powershell
python lora.py --subjects all
```

Fine-tune selected subjects:

```powershell
python lora.py --subjects 1,2,3
```

Useful options:

```powershell
python lora.py --epochs 20
python lora.py --batch-size 32
python lora.py --lr 0.001
python lora.py --checkpoint-path best_model.pt
python lora.py --output-dir finetune
```

LoRA defaults:

```text
rank = 4
alpha = 8.0
dropout = 0.1
target layers = linear1, linear2, attention_pool, classifier
```

After fine-tuning, LoRA weights are merged into the base Linear layers. The
saved subject models are normal `EEGTransformer` state dicts, not standalone
adapter-only files.


## Typical Full Run

```powershell
python moabb_data.py
python moabb_train.py --epochs 20 --checkpoint-path best_model.pt
python lora.py --subjects all --checkpoint-path best_model.pt
```
