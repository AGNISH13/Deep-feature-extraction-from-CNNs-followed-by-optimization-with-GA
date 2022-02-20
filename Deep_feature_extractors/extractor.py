from Deep_feature_extractors import googlenet,resnet18,vgg19
import numpy as np

def feature_extractor(folder_path, ext, out_classes):

    if ext == 'googlenet':

        df5, df6 = googlenet.model(folder_path, out_classes)
        array_train = np.asarray(df5)
        array_val = array_val = np.asarray(df6)
        return array_train, array_val

    elif ext == 'vgg':
        
        df5, df6 = vgg19.model(folder_path, out_classes)
        array_train = np.asarray(df5)
        array_val = array_val = np.asarray(df6)
        return array_train, array_val

    else:
        
        df5, df6 = resnet18.model(folder_path, out_classes)
        array_train = np.asarray(df5)
        array_val = array_val = np.asarray(df6)
        return array_train, array_val