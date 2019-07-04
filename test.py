import pandas as pda
def test():
        X = [[0, 0], [1, 1]]
        y = [0, 1]
        clf = svm.SVC(gamma='scale')
        clf.fit(X, y)  
        pred = clf.predict([[2,2],[6,7]])
        print(pred)
        '''
        a = [[1,2,3,0,0],[4,5,6,0,0],[7,8,9,0,0]]
        #a = a.to_numpy
        a = np.array(a)
        #b = a[2:9]
        b = a[1:4,1:6]
        X = [a,a]
        #print(X[0])
        '''
        '''
        X,y = make_classification(n_samples=1000, n_features=25, n_informative=3,
                           n_redundant=2, n_repeated=0, n_classes=8,
                           n_clusters_per_class=1, random_state=0)
        svc = SVC(kernel="linear")
        rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
        rfecv.fit(X,y)
        ek = rfecv.ranking_
        #print(np.shape(X),np.shape(y))
        '''
        '''
        #print("Optimal number of features : %d" % rfecv.n_features_)
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        pca = PCA(n_components=2)
        pca.fit(X)
        '''
