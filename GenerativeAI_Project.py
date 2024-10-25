diffusion_transformer/
│
├── models/
│   ├── encoders.py         # Modality-specific encoders
│   ├── attention.py        # Multimodal attention block
│   ├── diffusion.py        # Diffusion process code
│   └── model.py            # Complete Diffusion Transformer model
│
├── datasets/
│   └── sen12ms.py          # SEN12MS dataset loader
│
├── train.py                # Training and evaluation loop
├── utils.py                # Utility functions (e.g., metrics)
└── main.py                 # Main script to run the experiments

