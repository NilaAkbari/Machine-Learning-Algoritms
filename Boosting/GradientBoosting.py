from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


#خواندن داده ها
boston = load_boston()
x = pd.DataFrame(boston.data, columns = boston.feature_names)
y = pd.Series(boston.target)

#جدا کزدن داده ها به دو دسته آموزشي و آزمايشي
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#يک رگرسو براي بوست گراديان مي سازيم
gradientregressor = GradientBoostingRegressor(max_depth=2, n_estimators =3, learning_rate = 1.0)

#رگرسور را آموزش مي دهيم
model = gradientregressor.fit(x_train, y_train)

#پاسخ داده هاي تست را پيشبيني مي کنيم
y_pred = model.predict(x_test)
#چاپ ميزان صحت
print(r2_score(y_pred, y_test))


feature_importance = model.feature_importances_

#اهميت را به ماکسيمم اهميت نسبت مي دهيم

feature_importance = 100.0*(feature_importance/feature_importance.max())
sorted_index = np.argsort(feature_importance)
pos = np.arange(sorted_index.shape[0]) +0.5
plt.barh(pos, feature_importance[sorted_index], align = 'center')
plt.yticks(pos, boston.feature_names[sorted_index])
plt.xlabel('اهميت نسبي')
plt.title('اهميت متغير')
plt.show()



#بهتر کردن

from sklearn.model_selection import GridSearchCV
LR = {'learning_rate':[0.15,0.1,0.10,0.05], 'n_estimators':[100,150,200,250]}
tuning = GridSearchCV(estimator = GradientBoostingRegressor(),
                      param_grid = LR, scoring = 'r2')
tuning.fit(x_train,y_train)
print(tuning.best_params_, tuning.best_score_)



