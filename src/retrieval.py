import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision.models import ViT_B_16_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

# Plot settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 11


def parse_args(args):
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "vit_b_16"], help="Model name")
    parser.add_argument("--dataset-dir", type=str, default="./dataset", help="Dataset directory")
    parser.add_argument("--target-dir", type=str, default="./query", help="Target image path or directory")
    parser.add_argument("--output-dir", type=str, default="./figures/retrieval", help="Output directory")

    args = parser.parse_args(args)
    return args


def get_image_paths(input_dir, extensions = ("jpg", "jpeg", "png", "bmp")):
    """
    Get image paths from the given directory
    """
    pattern = f"{input_dir}/**/*"
    img_paths = []
    for extension in extensions:
        img_paths.extend(glob.glob(f"{pattern}.{extension}", recursive=True))

    if not img_paths:
        raise FileNotFoundError(f"No images found in {input_dir}. Supported formats are: {', '.join(extensions)}")

    return img_paths


def extract_features(model, dataset_dir, trans, device):
    """
    Extract features from the dataset
    """
    print('===> Preparing image data..')
    dataset = ImageFolder(dataset_dir, transform=trans)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("===> Extracting features..")
    feature_vectors = []
    image_paths = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Extracting features"):
            inputs = inputs.to(device)
            features = model(inputs)
            feature_vectors.append(features.cpu().numpy())
            image_paths.append(dataloader.dataset.samples[len(feature_vectors) - 1][0])

    feature_vectors = np.vstack(feature_vectors)
    image_paths = np.array(image_paths)
    return feature_vectors, image_paths


def cosine_similarity(input_feature, feature_vectors):
    """
    Calculate cosine similarity between the input feature and the feature vectors
    """
    input_feature = input_feature / np.linalg.norm(input_feature)
    feature_vectors = feature_vectors / np.linalg.norm(feature_vectors, axis=1, keepdims=True)
    similarities = np.dot(feature_vectors, input_feature.T)
    return similarities.flatten()


def euclidean_similarity(input_feature, feature_vectors):
    """
    Calculate similarity based on Euclidean distance between the input feature and the feature vectors
    """
    input_feature = input_feature / np.linalg.norm(input_feature)
    feature_vectors = feature_vectors / np.linalg.norm(feature_vectors, axis=1, keepdims=True)
    distances = np.linalg.norm(feature_vectors - input_feature, axis=1)
    similarities = 1 / (1 + distances)
    return similarities


def main(args):
    args = parse_args(args)

    # Load model
    print("==> Loading model..")
    print(f"Model: {args.model}")
    if args.model == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    elif args.model == "vit_b_16":
        model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()  # Remove the classification head
    model.eval()

    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    # Data pre-processing
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Extract features
    feature_path = f"{args.dataset_dir}/features_{args.model}.npy"
    image_paths_path = f"{args.dataset_dir}/image_paths_{args.model}.npy"
    if not os.path.exists(feature_path) or not os.path.exists(image_paths_path):
        features, image_paths = extract_features(model, args.dataset_dir, trans, device)
        print('===> Saving features..')
        np.save(feature_path, features)
        np.save(image_paths_path, image_paths)
    else:
        print('===> Loading features..')
        features = np.load(feature_path)
        image_paths = np.load(image_paths_path)

    # Get target image paths
    if os.path.isfile(args.target_dir):
        target_paths = [args.target_dir]
    else:
        target_paths = get_image_paths(args.target_dir)

    for  target_path in target_paths:
        print()
        print(f"Query Image: {target_path}")

        # Preprocess the target image
        target_img = Image.open(target_path).convert('RGB')
        target_img = trans(target_img).unsqueeze(0)
        target_img = target_img.to(device)

        # Extract features from the target image
        with torch.no_grad():
            input_feature = model(target_img).cpu().numpy()

        # Calculate cosine similarity
        similarities = cosine_similarity(input_feature, features)
        # similarities = euclidean_similarity(input_feature, features)

        # Get top-5 similar images
        top5_indices = np.argsort(similarities)[::-1][:5]
        print("Top-5 Similar Indices:", top5_indices)
        top5_similarities = similarities[top5_indices]
        top5_image_paths = image_paths[top5_indices]
        print("Top-5 Similar Images:")
        for i, (path, sim) in enumerate(zip(top5_image_paths, top5_similarities)):
            print(f"{i + 1}: {path}, Similarity: {sim:.4f}")

        # Visualize the results
        plt.figure(figsize=(10, 2.5))
        plt.subplot(1, 6, 1)
        query_img = plt.imread(target_path)
        plt.imshow(query_img)
        plt.title("Query Image\n" + target_path, fontsize=9, color='red')
        plt.axis("off")
        for i, path in enumerate(top5_image_paths):
            plt.subplot(1, 6, i + 2)
            img = plt.imread(path)
            plt.imshow(img)
            plt.title(f"Rank {i + 1}\nSimilarity: {top5_similarities[i]:.4f}\n{path}", fontsize=9)
            plt.axis("off")
        plt.tight_layout()
        save_path = f"{args.output_dir}/retrieval_{args.model}_{os.path.basename(target_path)}"
        plt.savefig(save_path, dpi=300)
        print("Results saved to:", save_path)

if __name__ == '__main__':
    main(sys.argv[1:])