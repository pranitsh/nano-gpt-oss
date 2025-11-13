# Nano-GPT-OSS Language Model

**An open-source transformer that balances full-context and sliding-window attention for efficient, scalable LLM training and inference.**

<a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch"></a>
<a href="https://huggingface.co"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FFC107?logo=hugging%20face&logoColor=black" alt="Hugging Face"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>

## Key Improvements of GPT-OSS over GPT-2

### 🏗️ Architecture Enhancements
- **Mixture of Experts (MoE) in MLP** with a Router → Sparse experts active per token (big model capacity, low active FLOPs)
- **Gated Router** → Token-dependent routing to experts (shown inside MoE block)
- **SwiGLU Feed-Forward (FFN) modules** → Modern activation in FFN instead of GELU
- **Grouped Query Attention + RoPE** → Alternate attention that supports longer context and stable queries
- **Sliding Window Attention** → Efficient attention pattern that reduces computation while maintaining context
- **Sink Slots in Attention** → Learned aggregation slots for global context stability
- **RMSNorm** → More stable normalization layer

### 📊 Performance Improvements
- **Lower Training Loss** → Better convergence during training
- **Lower Validation Loss** → Better generalization to unseen data
- **Lower Memory Usage** → More efficient memory usage during training and inference
- **Lower Disk Space** → More efficient disk space usage during training and inference
- **Lower Inference Time** → Faster inference time during inference

## Dependencies
- [pytorch](https://pytorch.org) <3
-  `datasets` for huggingface datasets <3 (for loading datasets)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3
-  `ipywidgets` for optional jupyter notebook support 

## 📊 Dataset and Format

TinyStories can be found at [HuggingFace Datasets](https://huggingface.co/datasets/roneneldan/TinyStories).

### Data Fields:

Each story entry contains:

- `story`: The main story text
<details>
<summary>📝 Click to see example story</summary>

**Story:**

```
Once upon a time, there was a big, red ball that could bounce very high...
```

\[Rest of the example story\]

</details>

## Training Procedure

```bash
git clone https://github.com/VizuaraAI/nano-gpt-oss
cd nano-gpt-oss
pip install -r requirements.txt
python train.py
```

## Validation Loss per Step

```txt
warnings.warn(
Ep 1 (Step 000000): Train loss 12.325, Val loss 12.350
✅ Saved new best model with val_loss=12.350
  4% 150/4230 [06:02<2:26:59,  2.16s/it]Ep 1 (Step 000150): Train loss 4.319, Val loss 4.003
✅ Saved new best model with val_loss=4.003
  7% 300/4230 [12:16<2:21:36,  2.16s/it]Ep 1 (Step 000300): Train loss 3.853, Val loss 3.766
✅ Saved new best model with val_loss=3.766
 11% 450/4230 [18:18<2:16:48,  2.17s/it]Ep 1 (Step 000450): Train loss 3.784, Val loss 3.516
✅ Saved new best model with val_loss=3.516
 14% 600/4230 [24:26<2:11:35,  2.18s/it]Ep 1 (Step 000600): Train loss 3.619, Val loss 3.325
✅ Saved new best model with val_loss=3.325
 18% 750/4230 [30:32<2:06:03,  2.17s/it]Ep 1 (Step 000750): Train loss 3.444, Val loss 3.197
✅ Saved new best model with val_loss=3.197
 21% 900/4230 [36:36<2:00:24,  2.17s/it]Ep 1 (Step 000900): Train loss 3.341, Val loss 3.144
✅ Saved new best model with val_loss=3.144
 25% 1050/4230 [42:53<1:53:34,  2.14s/it]Ep 1 (Step 001050): Train loss 3.266, Val loss 3.075
✅ Saved new best model with val_loss=3.075
 28% 1200/4230 [49:01<1:48:23,  2.15s/it]Ep 1 (Step 001200): Train loss 2.944, Val loss 2.916
✅ Saved new best model with val_loss=2.916
 32% 1350/4230 [55:06<1:43:01,  2.15s/it]Ep 1 (Step 001350): Train loss 3.138, Val loss 2.850
✅ Saved new best model with val_loss=2.850
 35% 1500/4230 [1:01:15<1:37:45,  2.15s/it]Ep 1 (Step 001500): Train loss 2.991, Val loss 2.788
✅ Saved new best model with val_loss=2.788
 39% 1650/4230 [1:07:14<1:33:13,  2.17s/it]Ep 1 (Step 001650): Train loss 3.084, Val loss 2.759
✅ Saved new best model with val_loss=2.759
 43% 1800/4230 [1:13:32<1:28:07,  2.18s/it]Ep 1 (Step 001800): Train loss 2.866, Val loss 2.716
✅ Saved new best model with val_loss=2.716
 46% 1950/4230 [1:19:50<1:22:15,  2.16s/it]Ep 1 (Step 001950): Train loss 3.050, Val loss 2.638
✅ Saved new best model with val_loss=2.638
 50% 2100/4230 [1:25:54<1:17:21,  2.18s/it]Ep 1 (Step 002100): Train loss 2.575, Val loss 2.594
✅ Saved new best model with val_loss=2.594
 53% 2250/4230 [1:32:02<1:11:41,  2.17s/it]Ep 1 (Step 002250): Train loss 2.628, Val loss 2.569
✅ Saved new best model with val_loss=2.569
 57% 2400/4230 [1:38:26<1:05:57,  2.16s/it]Ep 1 (Step 002400): Train loss 2.666, Val loss 2.600
 60% 2550/4230 [1:44:03<1:01:44,  2.21s/it]Ep 1 (Step 002550): Train loss 2.872, Val loss 2.553
✅ Saved new best model with val_loss=2.553
 64% 2700/4230 [1:49:55<54:39,  2.14s/it]Ep 1 (Step 002700): Train loss 2.841, Val loss 2.475
✅ Saved new best model with val_loss=2.475
 67% 2850/4230 [1:55:57<49:32,  2.15s/it]Ep 1 (Step 002850): Train loss 2.647, Val loss 2.459
✅ Saved new best model with val_loss=2.459
 71% 3000/4230 [2:02:06<44:14,  2.16s/it]Ep 1 (Step 003000): Train loss 2.759, Val loss 2.450
✅ Saved new best model with val_loss=2.450
 74% 3150/4230 [2:08:16<38:53,  2.16s/it]Ep 1 (Step 003150): Train loss 2.572, Val loss 2.462
 78% 3300/4230 [2:13:53<33:42,  2.17s/it]Ep 1 (Step 003300): Train loss 2.319, Val loss 2.381
✅ Saved new best model with val_loss=2.381
 82% 3450/4230 [2:19:55<28:05,  2.16s/it]Ep 1 (Step 003450): Train loss 2.697, Val loss 2.350
✅ Saved new best model with val_loss=2.350
 85% 3600/4230 [2:25:57<23:01,  2.19s/it]Ep 1 (Step 003600): Train loss 2.464, Val loss 2.347
✅ Saved new best model with val_loss=2.347
 89% 3750/4230 [2:31:56<17:16,  2.16s/it]Ep 1 (Step 003750): Train loss 2.556, Val loss 2.353
 92% 3900/4230 [2:37:30<11:53,  2.16s/it]Ep 1 (Step 003900): Train loss 2.728, Val loss 2.322
✅ Saved new best model with val_loss=2.322
 96% 4050/4230 [2:43:30<06:32,  2.18s/it]Ep 1 (Step 004050): Train loss 2.278, Val loss 2.278
✅ Saved new best model with val_loss=2.278
 99% 4200/4230 [2:50:47<01:04,  2.16s/it]Ep 1 (Step 004200): Train loss 2.370, Val loss 2.263
✅ Saved new best model with val_loss=2.263
100% 4230/4230 [2:53:24<00:00,  2.46s/it]
```

# First Epoch

Input: "a fast driver named Tim went for"
Output: "a walk. Tim saw a big, scary toy car. He wanted to play with, but he didn't know. So, Tim went to visit with his friend, Tim.

Tim said, "Hi, Tim! Do you want to play with me?" Tim walked to the car and asked, "Yes, please!" His friend Tim. Tim tried to catch the car, but he couldn't too fast.

Tim and his friend played with the toy car for a while, Tim found a toy car."

# Models

Models available per request.
