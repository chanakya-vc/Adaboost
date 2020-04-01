# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:19:31 2020

@author: Chanakya-vc
"""

from adaboost import stump
import joblib
from adaboost import Adaboost_algo
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import cv2
import os
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
path=r"C:/Users/vaibh/Documents/Notes_Hw/Sem2/CV/Project/faces"
# =============================================================================
# Read data and test
# =============================================================================
train_face=path +r"/train/face"
train_nface=path +r"/train/non-face"
# test_face=path +r"/test/face"
# test_nface=path +r"/test/non-face"
training_face=[]
training_nface=[]
# testing_face=[]
# testing_nface=[]
# training=[]
# test=[]

children=os.listdir(train_face)
for child in children:
    img=cv2.imread(train_face+ '/' + child,0)
    img=cv2.resize(img,(20,20))
    # img=img.reshape(100)
    training_face.append(img)
training_face=np.array(training_face)

children=os.listdir(train_nface)
for child in children:
    img=cv2.imread(train_nface+ '/' + child,0)
    img=cv2.resize(img,(20,20))
    # img=img.reshape(100)
#    img=img.reshape(len(img)**2)
    training_nface.append(img)
training_nface=np.array(training_nface)

# children=os.listdir(test_face)
# for child in children:
#     img=cv2.imread(test_face+ '/' + child,0)
#     img=cv2.resize(img,(20,20))
#     # img=img.reshape(100)
# #    img=img.reshape(len(img)**2)
#     testing_face.append(img)
# testing_face=np.array(testing_face)

# children=os.listdir(test_nface)
# for child in children:
#     img=cv2.imread(test_nface+ '/' + child,0)
#     img=cv2.resize(img,(20,20))
#     # img=img.reshape(100)
# #    img=img.reshape(len(img)**2)
#     testing_nface.append(img)
# testing_nface=np.array(testing_face)

# =============================================================================
# normalize data
# =============================================================================

def  normalize(data):
    scale=MinMaxScaler()
    scale.fit(data)
    data=scale.transform(data)
    return data
# =============================================================================
# Choose top 10 features without adaboost and also return their coordinates
# =============================================================================
def haar_without_ada(train_harr,labels,feature_maps):
    coord=[]
    for feature in feature_maps:
        feat_coord,_=haar_like_feature_coord(20, 20, feature)
        coord.append(feat_coord)
    error=[]
    indicator=np.ones((np.size(train_harr,0)))
    for j in range(np.size(train_harr,1)):
        threshold=np.mean(train_harr[:,j])
        predictions=np.ones(np.size(train_harr,0))
        for k,feats in enumerate(train_harr[:,j]):
            if(feats< threshold):
                predictions[k]=-1
        error_thresh=sum(indicator[predictions!=labels])
        error.append((error_thresh))
    error=np.array(error)
    # Get 10 most minimum values of error along with their coords
    indices=np.argsort(error)
    return coord, indices
    
        
        

def haar_features(train, test, feature_maps):
    train_harr=[]
    test_harr=[]
    for i in train:
        img_ii=integral_image(i)
        feature=haar_like_feature(img_ii,0,0,20,20,feature_maps)
        train_harr.append(feature)
    for i in test:
        img_ii=integral_image(i)
        feature=haar_like_feature(img_ii,0,0,20,20,feature_maps)
        test_harr.append(feature)
    train_harr=np.array(train_harr)
    test_harr=np.array(test_harr)
    return train_harr, test_harr

def accuracy(predictions,labels):
    count=0
    for i in range(np.size(predictions,0)):
        if(predictions[i]==labels[i]):
            count+=1
    return (count/np.size(predictions))*100
if __name__=="__main__":
    training_face=np.array(training_face[0:600,:,:])
    training_nface=np.array(training_nface[0:300,:,:])
    # testing_face=np.array(testing_face[0:50,:,:])
    # testing_nface=np.array(testing_nface[0:30,:,:])
    
    # train_imgs=np.vstack((training_face,training_nface))
    # test_images=np.vstack((testing_face,testing_nface))
    # labels_train=np.concatenate((np.ones(np.size(training_face,0)),  -1*np.ones(np.size(training_nface,0))))
    # labels_test=np.concatenate((np.ones(np.size(testing_face,0)),  -1*np.ones(np.size(testing_nface,0))))
    train_imgs=np.vstack((training_face[0:500,:,:],training_nface[0:250,:,:]))
    test_images=np.vstack((training_face[501:500,:,:],training_nface[251:300,:,:]))
    labels_train=np.concatenate((np.ones(np.size(training_face[0:500,:,:],0)),  -1*np.ones(np.size(training_nface[0:250,:,:],0))))
    labels_test=np.concatenate((np.ones(np.size(training_face[501:500,:,:],0)),  -1*np.ones(np.size(training_face[251:300,:,:],0))))
    
   
    
    feature_maps=('type-2-x','type-3-y')
    train_harr, test_harr=haar_features(train_imgs,test_images,feature_maps)        
    print("=====Feature Extraction Completed=====")
    
    classifier=Adaboost_algo(7)
    ensemble_ada=classifier.train(train_harr,labels_train)
    filename='adaboost_model'
    joblib.dump(ensemble_ada,filename)
    
    predictions=classifier.predict(test_harr)
    score=accuracy(predictions,labels_test)
    print(score)
    
    # indices, coord=haar_without_ada(train_harr,labels_train,feature_maps)
    

    
    