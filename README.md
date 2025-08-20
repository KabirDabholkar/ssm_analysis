# Student-Teacher Experiments with Dynamax State-Space Models

This repository contains the code and experiments for the paper:

**"When predict can also explain: few-shot prediction to select better neural latents"**  
*Kabir Dabholkar, Omri Barak*  
[arXiv:2405.14425](https://arxiv.org/abs/2405.14425)


## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/KabirDabholkar/ssm_analysis.git
   cd ssm_analysis
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```


## Usage

The experiments can be run using the configuration files in the `configs/` directory. See the individual configuration files for specific experiment setups.

### Required environment variable

Before running `main.py`, set the base directory where results will be written. To keep outputs tracked within this repository, point it to the repo root.

For zsh on macOS:
```bash
export RESULT_BASE_PATH="$HOME/Documents/code/ssm_analysis"
```

You can add the above line to your `~/.zshrc` to persist it across shells. Verify with:
```bash
echo $RESULT_BASE_PATH
```

Then run your experiments, e.g.:
```bash
python main.py
```

## Citation


```bibtex
@article{dabholkar2024predict,
  title={When predict can also explain: few-shot prediction to select better neural latents},
  author={Dabholkar, Kabir and Barak, Omri},
  journal={arXiv preprint arXiv:2405.14425},
  year={2024}
}
```