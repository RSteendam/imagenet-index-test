import torch
import pickle
import os

FEATURE_FILES = [
    "resnext101_32x48d-avg-logits.pickle", 
    "resnext101_32x48d-max-logits.pickle", 
    "senet154-avg-logits.pickle",
    "senet154-max-logits.pickle",    
    ]
FEATURE_KEY = "a4209223a8"
IMAGENET_FILE = os.path.join("imagenet", "imagenet_classes.txt")
DIDEMO_FILE = os.path.join("captions", "didemo-captions.pkl")


def main():
    print("TESTING IMAGENET CLASSES ON RESNEXT AND SENET")
    print()

    
    captions = pickle.load(open(DIDEMO_FILE, 'rb'))

    with open(IMAGENET_FILE, "r") as f:
        categories = [s.strip() for s in f.readlines()]   

    print(f"clip: {FEATURE_KEY}")
    print()

    print("SENTENCES:")
    for sentence in captions[FEATURE_KEY]:
        print(" ".join(sentence))
    print()

    print("PREDICTIONS:")
    for feature_file in FEATURE_FILES:
        print(f"model: {feature_file}")
        test_feature(feature_file, FEATURE_KEY, categories)
    print("END")

def test_feature(feature_file, key, imagenet_categories):
    feature_path = os.path.join("features", feature_file)
    features = pickle.load(open(feature_path, 'rb'))

    output = torch.from_numpy(features[key]).float()
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    for i in range(top5_prob.size(0)):
        print(imagenet_categories[top5_catid[i]], top5_prob[i].item())
    print()

if __name__ == '__main__':
    main()