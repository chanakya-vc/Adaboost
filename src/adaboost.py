# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 21:08:15 2020

@author: Chanakya-vc
"""
import numpy as np
import math
# =============================================================================
# Adaboost Functionality
# =============================================================================
class stump:
    def __init__(self,alpha,index,threshold,polarity=1):
        self.alpha=alpha
        self.polarity=polarity
        self.index=index
        self.threshold=threshold


class Adaboost_algo:
    def __init__(self,num_trees):
        self.num_trees=num_trees
        self.ensemble=[]
    def train(self,features,labels):
        weights=(1/np.size(features,0))*np.ones((np.size(features,0)))
        for i in range(self.num_trees):
            
            tree=stump(-1,-1,-1)
            error_thresh_min=float('inf')
            threshold_min=0 
# =============================================================================
#             Find the feature that gives the minimum error. For each such feature find the threshold  value
# =============================================================================
            for j in range(np.size(features,1)):                 
                # for threshold in features[:,j]:
                 threshold=np.mean(features[:,j])
                 predictions=np.ones(np.size(features,0))
                 polarity_flag=1
                 for k,feats in enumerate(features[:,j]):
                     if(feats< threshold):
                         predictions[k]=-1
                 error_thresh=sum(weights[predictions!=labels])
# =============================================================================
#                      Change the polarity if the error>0.5 so that the tree predicts the opposite for feat<threshold
# =============================================================================
                 if error_thresh> 0.5:
                     error_thresh=1-error_thresh
                     polarity_flag=-1
                 if (error_thresh < error_thresh_min):
                     error_thresh_min=error_thresh
                     threshold_min=threshold
                     polarity=polarity_flag
                     index=j
                     predictions_save=predictions

# =============================================================================
#             Save the stump with the minimum error along with the threshold value
# =============================================================================
            tree.alpha=0.5*math.log((1-error_thresh_min)/(error_thresh_min + 1e-9))
            tree.threshold=threshold_min
            tree.polarity=polarity
            tree.index= index   
# =============================================================================
#               Change the values of the weights for the next stump                 
# =============================================================================
            predictions=tree.polarity*predictions_save
            # predictions=np.ones(np.size(features,0)) 
            # for k,feat in enumerate(features[:,tree.index]):
            #     # if (feat < tree.threshold):
            #     #     predictions[k]=tree.polarity*(-1)
            #     predictions[feat<tree.threshold]=-tree.polarity
            #     predictions[feat>tree.threshold]=tree.polarity
            # predictions[features[:,tree.index]<tree.threshold]=-tree.polarity
            # predictions[features[:,tree.index]>=tree.threshold]=tree.polarity
            weights[predictions!=labels]*=np.exp(tree.alpha)
            weights[predictions==labels]*=np.exp(-tree.alpha) 
            # weights*=np.exp(-tree.alpha*labels*predictions)
            weights=weights/sum(weights)
            self.ensemble.append(tree)
            print("===Stump"+str(i)+"trained")
        ensemble=self.ensemble
        return ensemble
    
    def predict(self,test):
        ensemble=self.ensemble
        preds=np.zeros(np.size(test,0))
        for tree in ensemble:
            weak_preds=np.ones(np.shape(preds))
            # for k,feat in enumerate(test[:,tree.index]):
            #     weak_preds[feat<tree.threshold]=-tree.polarity
            #     weak_preds[feat>tree.threshold]=tree.polarity
            weak_preds[test[:,tree.index]<tree.threshold]=-tree.polarity
            weak_preds[test[:,tree.index]>=tree.threshold]=tree.polarity
            preds+=tree.alpha*weak_preds
        return np.sign(preds)
            
        
            
    
        
        
        
        
                    
                            
                    
                    
                    
                    
                
            
            
            
            
    
    
    
        
        