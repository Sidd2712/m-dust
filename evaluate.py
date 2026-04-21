# evaluate.py
import tensorflow as tf
import argparse
from models.layers import MSEEMLayer, FrequencyFilterLayer, DAIMBlock
from utils.losses import MDUSTLoss
from train import build_dataset # Reusing the data pipeline from train.py

def evaluate_model(model_path, test_data_path):
    print(f"Loading model from {model_path}...")
    
    # Define custom objects required to load the saved model
    custom_objects = {
        'MSEEMLayer': MSEEMLayer,
        'FrequencyFilterLayer': FrequencyFilterLayer,
        'DAIMBlock': DAIMBlock,
        'MDUSTLoss': MDUSTLoss
    }
    
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    print(f"Preparing dataset from {test_data_path}...")
    test_ds = build_dataset(test_data_path, batch_size=32)
    
    print("Evaluating Cross-Domain Performance...")
    results = model.evaluate(test_ds, return_dict=True)
    
    print("\n--- Evaluation Results ---")
    print(f"Classification Accuracy: {results['cls_accuracy']:.4f}")
    print(f"Classification AUC:      {results['cls_auc']:.4f}")
    print(f"Segmentation Accuracy:   {results['seg_accuracy']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate M-DUST Model")
    parser.add_argument('--model', type=str, default='mdust_best_weights.keras', help='Path to saved model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to test dataset (e.g., data/Celeb-DF/test)')
    args = parser.parse_args()
    
    evaluate_model(args.model, args.dataset)