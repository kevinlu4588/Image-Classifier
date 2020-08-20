import argparse
import predict_functions
import json
parser = argparse.ArgumentParser(
    description='Please enter desired hyperparamters:',
)
parser.add_argument("--image_path", dest = "image_path", action = "store", default = "ImageClassifier/flowers/test/10/image_07117.jpg")
parser.add_argument("--checkpoint_path", dest = "checkpoint_path", action = "store", default = "checkpoint.pth")
parser.add_argument("--top_classes", dest = "n_topk", action = "store", default = 3)
parser.add_argument("--gpu", dest = "gpu", action = "store", default = False, type = bool)
parser.add_argument("--category_names", dest = "category_file", action = "store", default = None, type = str)

args = parser.parse_args()

model = predict_functions.load_checkpoint(args.checkpoint_path)
probs, classes = predict_functions.predict(args.image_path, model, args.n_topk, args.gpu) 
name_or_class = ""
if args.category_file != None:
    with open(args.category_file, 'r') as f:
        cat_to_name = json.load(f)
        names = [cat_to_name[image_class] for image_class in classes]
        name_or_class = " Name"
else:
    names = classes
    name_or_class = " Class"
        
print(f"Top {args.n_topk} Flower{name_or_class} Probablities:\n")
for i in range(args.n_topk):
        print(f"{name_or_class}: {names[i]} Probability: {probs[i]*100}%\n")
