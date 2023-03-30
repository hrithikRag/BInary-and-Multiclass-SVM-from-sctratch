import numpy as np
from tqdm import tqdm


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
    
    def fit(self, X) -> None:
        # fit the PCA model
        X_avg = X - np.mean(X, axis=0)
        cov_mat = np.cov(X_avg, rowvar=False)
        eign_val, eign_vec = np.linalg.eigh(cov_mat)
        sorted_index = np.argsort(eign_val)[::-1]
        sorted_eign_val = eign_val[sorted_index]
        sorted_eign_vec = eign_vec[:,sorted_index]
        self.components = sorted_eign_vec[:, 0:self.n_components]
    
    def transform(self, X) -> np.ndarray:
        # transform the data
        X_avg = X - np.mean(X, axis=0)
        projected_data = np.dot(self.components.T, X_avg.T).T

        return projected_data


    def fit_transform(self, X) -> np.ndarray:
        # fit the model and transform the data
        self.fit(X)
        return self.transform(X)


class SupportVectorModel:
    def __init__(self) -> None:
        self.w = None
        self.b = None
    
    def _initialize(self, X) -> None:
        # initialize the parameters
        self.row=X.shape[0]
        self.col=X.shape[1]
        self.w=np.zeros(self.col)
        self.b=0
        
    def pred_num(self,i,y_train):
        Y_label=np.where(y_train!=i,-1,1)
        return Y_label
    
    def fit(
            self, X, y, 
            learning_rate: float,
            num_iters: int,
            C: float = 1.0,
    ) -> None:
        self._initialize(X)
        
        # fit the SVM model using stochastic gradient descent
        for i in tqdm(range(1, num_iters + 1)):
            samples=59999
            x_indices=np.random.choice(samples, size=1, replace=True)
            x_rand=[]
            for i in x_indices:
                x_rand.append(list(X)[i])
            for i, x in enumerate(x_rand):
                if 1 - y[i]*(np.dot(x,self.w)+self.b) <=0 :
                    dw = 2*self.w
                    db = 0
                else:
                    dw = (2*self.w) - C*y[i]*x
                    db = -y[i]
                self.w = self.w - (learning_rate*dw)
                self.b = self.b - (learning_rate*db)

    
    def predict(self, X) -> np.ndarray:
        # make predictions for the given data
        result= np.dot(X,self.w)
        predicted_result=np.sign(result)
        predicted_labels=np.where(predicted_result <=-1,0,1)
        dist_from_plane=result/np.dot(self.w,self.w)
        return predicted_labels,dist_from_plane

    def accuracy_score(self, X, y) -> float:
        # compute the accuracy of the model (for debugging purposes)
        return np.mean(self.predict(X) == y)


class MultiClassSVM:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.models = []
        for i in range(self.num_classes):
            self.models.append(SupportVectorModel())
    
    
    def fit(self, X, y, C,learning_rate,num_iters) -> None:
        
        for i in range(10):
            self.models[i].fit(X, self.models[i].pred_num(i,y),learning_rate,num_iters)
            
        

    def predict(self, X) -> np.ndarray:
        # pass the data through all the 10 SVM models and return the class with the highest score
        final_pred=[]
        y_0_pred=[]
        y_1_pred=[]
        y_2_pred=[]
        y_3_pred=[]
        y_4_pred=[]
        y_5_pred=[]
        y_6_pred=[]
        y_7_pred=[]
        y_8_pred=[]
        y_9_pred=[]
        for x in X:
            dist_from_plane=[]
            temp_dist=[]
            labels=[]
    
            Y_test_pred,dist=self.models[0].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_0_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[1].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_1_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[2].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_2_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[3].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_3_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[4].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_4_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[5].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_5_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[6].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_6_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[7].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_7_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[8].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_8_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[9].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_9_pred.append(int(Y_test_pred))
    
            for i in range(10):
                count=0
                if labels[i]==1:
                    count+=1
                if dist_from_plane[i]<=0:
                    dist_from_plane[i]=abs(dist_from_plane[i])
    
            if count==0:
                final_pred.append(dist_from_plane.index(min(dist_from_plane)))
            elif count==1:
                final_pred.append(labels.index(1))
            else:
                for i in range(10):
                    if labels[i]==1:
                        temp_dist.append(dist_from_plane[i])
                final_pred.append(dist_from_plane.index(max(temp_dist)))
        
        return final_pred

    def accuracy_score(self, X, y) -> float:
        return np.mean(self.predict(X) == y)
    
    def recall_score(self, X, y) -> float:
        final_pred=[]
        y_0_pred=[]
        y_1_pred=[]
        y_2_pred=[]
        y_3_pred=[]
        y_4_pred=[]
        y_5_pred=[]
        y_6_pred=[]
        y_7_pred=[]
        y_8_pred=[]
        y_9_pred=[]
        for x in X:
            dist_from_plane=[]
            temp_dist=[]
            labels=[]
    
            Y_test_pred,dist=self.models[0].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_0_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[1].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_1_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[2].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_2_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[3].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_3_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[4].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_4_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[5].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_5_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[6].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_6_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[7].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_7_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[8].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_8_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[9].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_9_pred.append(int(Y_test_pred))
    
            for i in range(10):
                count=0
                if labels[i]==1:
                    count+=1
                if dist_from_plane[i]<=0:
                    dist_from_plane[i]=abs(dist_from_plane[i])
    
            if count==0:
                final_pred.append(dist_from_plane.index(min(dist_from_plane)))
            elif count==1:
                final_pred.append(labels.index(1))
            else:
                for i in range(10):
                    if labels[i]==1:
                        temp_dist.append(dist_from_plane[i])
                final_pred.append(dist_from_plane.index(max(temp_dist)))
        
        compiled_svm_pred=[y_0_pred,y_1_pred,y_2_pred,y_3_pred,y_4_pred,y_5_pred,y_6_pred,y_7_pred,y_8_pred,y_9_pred]
        recall=[]
        
        for i in range (10):
            temp_Y_test=np.where(y==i,1,0)
            true_pos=0
            actual_pos=1
            for j in range(len(y)):
                if temp_Y_test[j]==1:
                    actual_pos+=1
                if temp_Y_test[j]==compiled_svm_pred[i][j] and temp_Y_test[j]==1:
                    true_pos+=1
            recall_test=true_pos/actual_pos
            recall.append(recall_test)
        final_recall=np.array(recall).mean()
        return final_recall       
        
        
    
    def precision_score(self, X, y) -> float:
        final_pred=[]
        y_0_pred=[]
        y_1_pred=[]
        y_2_pred=[]
        y_3_pred=[]
        y_4_pred=[]
        y_5_pred=[]
        y_6_pred=[]
        y_7_pred=[]
        y_8_pred=[]
        y_9_pred=[]
        for x in X:
            dist_from_plane=[]
            temp_dist=[]
            labels=[]
    
            Y_test_pred,dist=self.models[0].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_0_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[1].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_1_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[2].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_2_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[3].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_3_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[4].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_4_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[5].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_5_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[6].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_6_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[7].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_7_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[8].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_8_pred.append(int(Y_test_pred))
    
            Y_test_pred,dist=self.models[9].predict(x)
            dist_from_plane.append(dist)
            labels.append(int(Y_test_pred))
            y_9_pred.append(int(Y_test_pred))
    
            for i in range(10):
                count=0
                if labels[i]==1:
                    count+=1
                if dist_from_plane[i]<=0:
                    dist_from_plane[i]=abs(dist_from_plane[i])
    
            if count==0:
                final_pred.append(dist_from_plane.index(min(dist_from_plane)))
            elif count==1:
                final_pred.append(labels.index(1))
            else:
                for i in range(10):
                    if labels[i]==1:
                        temp_dist.append(dist_from_plane[i])
                final_pred.append(dist_from_plane.index(max(temp_dist)))
        
        compiled_svm_pred=[y_0_pred,y_1_pred,y_2_pred,y_3_pred,y_4_pred,y_5_pred,y_6_pred,y_7_pred,y_8_pred,y_9_pred]
        precision=[]
        
        for i in range (10):
            temp_Y_test=np.where(y==i,1,1)
            true_pos=0
            total_pos=1
            for j in range(len(y)):
                if compiled_svm_pred[i][j]==1:
                    total_pos+=1
                if temp_Y_test[j]==compiled_svm_pred[i][j] and temp_Y_test[j]==1:
                    true_pos+=1
            precision_test=true_pos/total_pos
            precision.append(precision_test)
        final_precision=np.array(precision).mean()
        return final_precision        
        
            
    
    def f1_score(self, X, y) -> float:
        return 2 / ((1/self.precision_score(X, y)) + (1/self.recall_score(X, y)))