from sklearn import tree

clf_decision_tree = tree.DecisionTreeClassifier()
clf_decision_tree_regressor =  tree.DecisionTreeRegressor()

#[height,weight,shoe size]
X= [[181,80,44],[177,70,43],[160,60,38],[154,54,37],
    [166,65,40],[190,90,47],[175,64,39],[177,70,40],
    [159,55,37],[171,75,42],[181,85,43]
    ]
    
Y = ['male','male','female','female','male','male','female','male','male','female','male',]

clf_decision_tree = clf_decision_tree.fit(X,Y)
clf_decision_tree_regressor = clf_decision_tree_regressor.fit(X,Y)


predict_decision_tree = clf_decision_tree.predict([[190,70,43]])
predict_decision_tree_regressor = clf_decision_tree_regressor.predict([[190,70,43]])

print(predict_decision_tree)
print(predict_decision_tree_regressor)