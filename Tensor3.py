#تشخیص اعداد
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense , Dropout
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
model.add(Dense(60, activation='relu',input_dim = 25))
model.add(Dropout(0.2))
model.add(Dense(300,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
print(model.summary()) #نوشتن اطلاعات 
print(".......................................................")
#بهینه سازی و تابع هزینه
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#در نهایت فاز فیت کردن یا آموزش دادن
history = model.fit(x_train,y_train,
                    epochs=150,
                    batch_size=128 , validation_data=(x_test,y_test) , verbose=0) #آخری یعنی از هزار تا سمپل بیست درصد را نبین اینو به عنوان ارزیابی مدل استفاده کن 

import matplotlib.pyplot as plt

plt.plot(history.history['val_accuracy'])
plt.plot(history.history['accuracy'])
plt.legend(['val','train'])
plt.show() #رسم نمودار مقایسه دقت آموزش و ارزیابی
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.legend(['val','train'])
plt.show() #رسم نمودار مقایسه خطا آموزش و ارزیابی
#با رسم نمودارهای بالا میبینیم که  اورفیت رفع شده و دقت ارزیابی افزایش پیدا کرده
