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

### Live Mode

The default behavior of `main.py` is to run in **live mode**. In this case, the classifier will test each frame that passes through the camera.

```sh
python main.py
```

You can also tell `main.py` to run explicitly in live mode:

```sh
python main.py live
```

### Image Inputs

To detect bird/cockatiel instances from an image:

```sh
python main.py detect -i cockatiel.jpg
```

To run the classifier in live mode:

```sh
python main.py live
```

### Optimization (Optional)

To find optimal values for a specific dataset class:

```sh
python main.py optimize -d dataset/cockatiel
```

Note that this wouldn't affect the classifier's existing state, unless it is modified to use the values.
