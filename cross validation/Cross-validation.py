from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

# در قطعه اول کد در اینجا ، ما یک مجموعه داده به نام مجموعه داده داریم
# kfold را اضافه مي کنيم و به دوقسمت براي آموزش و تست تقسيم مي کنيم
# آن را پرينت مي کنيم
dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
kf = KFold(n_splits=2, shuffle=False)
for train, test in kf.split(dataset):
    print("%s %s" % (train, test))

#اکنون مي خواهيم از cross-validation استفاده کنيم
# ديتا ست وارد شده از sklearn را مي خوانيم
iris = load_iris()
X = iris.data
y = iris.target

print('')
knn = KNeighborsClassifier()

# تعداد بهينه همسايه ها را براي knn چک مي کنيم
for i in range(20):
    knn.n_neighbors = i+1
    print(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())

# دو مدلی را که می خواهیم با هم چک کنيم تنظیم کنید
knn.n_neighbors = 20
logreg = LogisticRegression()

# از cross_val_score برای گرفتن امتیاز دقیق دو مدل استفاده می کند
# cv = 10  امتياز را مي خوايممخفف چند برابر ما می خوا هد. در این حالت 10
# 'دقت' همان معیار ارزیابی است که ما انتخاب کرده ایم امتياز دهي يعني
# ما در نهایت برای بدست آوردن جواب بلافاصله از میانگین () استفاده می کنیم.
#بدون آن ، ما باید خودمان میانگین را محاسبه کنیم
#بر اساس پاسخ هایی که می توانیم ببينيم، مدلی را انتخاب کنیم که بهترین کارايي را داشته باشد.
print('')
print(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())
print(cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())

