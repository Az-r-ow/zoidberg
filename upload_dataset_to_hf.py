"""
  In this file we'll be processing and uploading the chest_xray dataset to Hugging Face
"""

from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="./datasets/chest_Xray")

dataset.push_to_hub("Az-r-ow/chest_xray")
