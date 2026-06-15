#تشخیص اعداد
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from dataset import load_hoda #یک دیتا ست که توسط پایتون نوشته شده است

x_train_orginal , y_train_orginal , x_test_orginal , y_test_orginal= load_hoda()
print(x_train_orginal.shape) #جواب میشود هزار سمپل در بیست و پنج فیجر 
print(x_test_orginal.shape)
#وای تست و وای ترین را به فرمت وان هات تبدیل میکنیم چون خروجی شبکه عصبی صفر و یک قبول میکند
y_train = tf.keras.utils.to_categorical(y_train_orginal)
y_test = tf.keras.utils.to_categorical(y_test_orginal)
print(y_train[:5]) #فرمت وان هات
#نرمال سازی دیتا برای اینکه در یک رنج بیاریم آن را بین 0 و 1
print(".......................................................")
x_test = x_test_orginal.astype('float32') #تمام فریمورک ها فلوت32 را ساپورت میکنند
x_train = x_train_orginal.astype('float32') # عدد 32 یعنی چهارتا هشت بیتی
x_train /= 255 #چون تصاویر بین صفر تا 255 است
x_test /= 255
# تا ایتجا ما دیتا را ساختیم
#از ایانجا میریم سراغ تعریف معماری مدل یا اتصالات شبکه
model = Sequential()
model.add(Dense(64, activation='relu',input_dim = 25))
model.add(Dense(10, activation='softmax'))
print(model.summary()) #نوشتن اطلاعات 
print(".......................................................")
#بهینه سازی و تابع هزینه
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
#در نهایت فاز فیت کردن یا آموزش دادن
history = model.fit(x_train,y_train,
                    epochs=100,
                    batch_size=64 , validation_split=0.2 , verbose=0) #آخری یعنی از هزار تا سمپل بیست درصد را نبین اینو به عنوان ارزیابی مدل استفاده کن 

import matplotlib.pyplot as plt

plt.plot(history.history['val_accuracy'])
plt.plot(history.history['accuracy'])
plt.legend(['val','train'])
plt.show() #رسم نمودار مقایسه دقت آموزش و ارزیابی
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.legend(['val','train'])
plt.show() #رسم نمودار مقایسه خطا آموزش و ارزیابی
#ارزیابی مدل روی داده های آزمون
loss , acc = model.evaluate(x_test , y_test)
print('\nTesting loss: %.2f, acc : %.2f%%'%(loss , acc)) #چاپ دقت ارزیابی
print("...........................................................")
#انجام چند پیش بینی
predict = model.predict(x_test)
print(predict)
print(".......................................................")
print(predict.shape) #دویست سمپل و ده نورون خروجی یا احتمالات دارم، برای هر سمپل ده احتمال یا خروجی 
print(".......................................................") 
print(predict[1]) #دیدن مقادیر احتمال سمپل اول
print(".......................................................")
print(predict[1].max()) #درصد اطمینان کلاس برنده رو میگوید
print(".......................................................")
print(predict[1].argmax()) #دیدن اینکه کلاس برنده کدوم بوده
print(".......................................................")
print(y_test[1])  #میبینیم عدد 1 یا همان برنده روی خانه دوم است
print(".......................................................")
plt.imshow(x_test[1].reshape(5,5) , cmap = "gray") #رسم سمپل برنده
plt.show()
#برای دیدن  اینکه کدوم دارای خطا است
print("Predict : ")
print(predict.argmax(axis = 1)) #دیدن هر برنده برای دویست سمپل در پیش بینی
print("True Lable :")
print(y_test.argmax(axis=1)) #دیدن هر برنده در دویست سمپل در وای های اصلی
#سپس نتیحه این دو را مقایسه
print(predict.argmax(axis = 1)==y_test.argmax(axis=1))
#شمارش تعداد درست
print(np.sum(predict.argmax(axis = 1)==y_test.argmax(axis=1)))
#جمع بالا یعنی درست ها رو یک بشما و غلط هاا رو صفر سپس جمع کن 
#دیدن نتایج به صورت درصد
print(np.mean(predict.argmax(axis = 1)==y_test.argmax(axis=1)))
#دیدن نتایج به صورت نمودار
from sklearn.metrics import confusion_matrix
plt.matshow(confusion_matrix(y_test.argmax(axis=1),predict.argmax(axis = 1)) , cmap='Blues') #باید به صورا برچسب بدهیم هر دو را یعنی مثل هم باشند
plt.xticks(range(10))
plt.yticks(range(10))
plt.xlabel("Predict")
plt.ylabel("True Lable")
plt.show()
#با توجه به نمودار برنامه ما عدد 7 را به اشتباه 2 تشخیص داده اما 2 را 7 تشخیص نداد و 2 را 1 به اشتباه تشخیص داده

