# Haar Cascade + SqueezeNet Classifier

## Installation

**Setup and activate Virtual Environment**

```sh
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux
.venv/bin/activate
```

**Install dependencies**

```sh
pip install -r requirements.txt
```

## Usage

To detect bird/cockatiel instances from an image:

```sh
python main.py detect -i cockatiel.jpg
```

Optional: to find optimal values for a specific dataset class:

```sh
python main.py optimize -d dataset/cockatiel
```
