# Gauge Detection and Reading Toolkit

This project provides scripts for generating synthetic data and building machine learning models for manual gauge reading and detection. It leverages Isaac Sim and Omniverse Replicator for data generation alongside FastRCNN for gauge detection.

## Components

- **gen_gauge.py**  
    Generates synthetic data for manual gauges using Isaac Sim and Omniverse Replicator.

- **gauge_net.py**  
    Constructs a FastRCNN model to read values from manual gauges.

- **gauge_detect.py**  
    Builds a FastRCNN model to detect manual gauges and their needles.

- **ROS2 worskpace**
   See documentation in the [`ros.ws`](ros.ws/README.md) folder 

## Getting Started

### Prerequisites

- Python 3.10
- Isaac Sim and Omniverse Replicator (for synthetic data generation)
- PyTorch (or your chosen deep learning framework for FastRCNN)

### Installation

1. Clone the repository:
     ```
     git clone https://github.com/hcltech-robotics/proj-cec-doggo-ai.git
     ```
2. Navigate to the project directory:
     ```
     cd proj-cec-doggo-ai
     ```
3. Install the necessary Python dependencies:
     ```
     python -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt
     ```

### Usage

#### Generating Synthetic Data

Run the following command to generate synthetic gauge data:
```
python gen_gauge.py
```

#### Training and Running the Models
- To train or run the gauge reading model:
    ```
    python gauge_net.py
    ```
- To train or run the gauge detection model:
    ```
    usage: gauge_detect.py [-h] --image_dir IMAGE_DIR [--model_path MODEL_PATH] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--epochs EPOCHS] [--finetune]
                       [--finetune_weights FINETUNE_WEIGHTS]

    Train gauge detection model.

    options:
    -h, --help            show this help message and exit
    --image_dir IMAGE_DIR
                            Directory containing gauge images.
    --model_path MODEL_PATH
                            Path to save the TorchScript model.
    --batch_size BATCH_SIZE
                            Batch size for training.
    --num_workers NUM_WORKERS
                            Number of DataLoader worker threads.
    --epochs EPOCHS       Number of epochs for training.
    --finetune            Load existing model weights for fine-tuning.
    --finetune_weights FINETUNE_WEIGHTS
                            Weights used for fine-tuning.

    ```