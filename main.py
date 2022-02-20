

from Deep_feature_extractors import extractor
from Genetic_Algorithm import GA
import argparse


parser = argparse.ArgumentParser(description = 'Application of Genetic Algorithm')
# Paths
parser.add_argument('-data','--data_folder',type=str, 
                    default = 'data', 
                    help = 'Path to data')
parser.add_argument('-classes','--num_classes',type=int, 
                    default = 2, 
                    help = 'Number of data classes')
parser.add_argument('-ext','--extractor_type',type=str, 
                    default = 'resnet', 
                    help = 'Choice of deep feature extractor')                    
parser.add_argument('-classif','--classifier_type',type=str, 
                    default = 'KNN', 
                    help = 'Choice of classifier for GA')



args = parser.parse_args()
folder_path = args.data_folder
out_classes = args.num_classes
ext = args.extractor_type
classif = args.classifier_type


print("Extracting deep features...")
print('\n'*2)
array_train, array_val = extractor.feature_extractor(folder_path, ext, out_classes)
print("Deep features extracted.")
print('\n'*2)
print('Applying Genetic Algorithm...')
print('\n'*2)
GA.algorithm(array_train, array_val, classif)