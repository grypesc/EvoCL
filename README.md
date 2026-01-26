# EvoCL repository

Create virtual environment and install dependencies
```bash
python3 -m venv venv && source venv/bin/activate
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install requirements.txt
```

To reproduce experiments when training from scratch, first train a model on 1st task using "gradient" script in ```scripts``` directory, e.g.:
```bash
bash scripts/gradient-mnist-5x2.sh
```
This will generate a model named "model_0.pth". Move it to the main directory and rename to mnist-5x2.pth.
Then run EvoCL:
```bash
bash scripts/evocl-mnist-5x2.sh
```

To reproduce results with the pretrained ViT, first download it:
```bash
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth
```
Then run:
```bash
bash scripts/evocl-vit-domainnet-6x10.sh
```
