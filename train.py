import argparse
import train_functions
parser = argparse.ArgumentParser(
    description='Please enter desired hyperparamters:',
)
parser.add_argument("--save_directory", dest="save_dir", action="store", default="checkpoint.pth")
parser.add_argument("--data_directory", dest = "data_dir", action = "store", default = "ImageClassifier/flowers")
parser.add_argument("--architecture", dest = "arch", action = "store", default = "vgg16")
parser.add_argument("--hidden_units", dest = "n_hidden", action = "store", default = "4096,1024", type = str, help = "Enter tuple (a,b) for #hidden input and out units in Linear Combination Layer #2")
parser.add_argument("--learning_rate", dest = "learnrate", action = "store", default = "0.001", type = float)
parser.add_argument("--epochs", dest = "epochs", action = "store", default = 5, type = int)
parser.add_argument("--gpu", dest = "gpu", action = "store", default = False, type = bool)
args = parser.parse_args()
args.n_hidden = eval(args.n_hidden)
#get dataloaders
traindata, trainloader, validloader, testloader = train_functions.process_data(args.data_dir)
#create and train model
model,optimizer = train_functions.create_model(args.arch, args.n_hidden, args.learnrate)
train_functions.train_model(model, trainloader, validloader, args.epochs, optimizer, args.gpu)
#save model
train_functions.save_model(model, args.save_dir, optimizer, traindata, args.epochs, args.arch)