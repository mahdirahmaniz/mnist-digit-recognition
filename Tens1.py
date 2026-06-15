import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


path = "F:/Software/Python Project/AI/CNN/dataset/mnist.npz.zip"


with np.load(path) as dataset:
    x_train = dataset['x_train']
    y_train = dataset['y_train']
    x_test = dataset['x_test']
    y_test = dataset['y_test']
x_train , x_test = x_train / 255.0 , x_test / 255.0
#دیدن یک سمپل
i = int(input("Pease enter a Number : "))
plt.imshow(x_train[i] , cmap='gray')
plt.title(y_train[i]); #نقطه سیمیکالون برای اینکه اطلاعات اصافی دیده نشود در تایتل
plt.show()
print("Train :" ,x_train.shape) #دیدن تعداد سمپل و فیچر آموزش 28*28
print(".......................................................")
print("Test :" , x_test.shape) #دیدن تعداد سمپل و فیچر تست 28*28
print(".......................................................")

#از ایانجا میریم سراغ تعریف معماری مدل یا اتصالات شبکه
#نسبت به کد قبل همه را یکجا فرامیخوانیم و جدا ایمپورا نمیکنیم
#تک تک لایه ها را به عنوان یک لست قرار میدهیم داخلش
model = tf.keras.models.Sequential([
 tf.keras.layers.Flatten(input_shape = (28,28)), #برای اینکه رشیپ رو خودش انجام بده و به یک وکتور تبدیلش کن
 tf.keras.layers.Dense(128 , activation='relu'),
 tf.keras.layers.Dropout(0.2), #سبب افزایش دقت میشود
 tf.keras.layers.Dense(10)  
])
#بهینه سازی
#چون در لایه آخر سافت مکس نزاشتیم پس فرام لاجیک ترو رو میزاریم در پایین
#چون وان هات نکردیم اسپارس میزاریم
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#آموزش
#پون دیتا خیلی بیشتر است تعداد ایپاک کم میکنیم پنج تا
model.fit(x_train , y_train , epochs = 5 , verbose=0)
#ارزیابی تصویر توسط تصاویری که مدل ندیده برای آموزی یا همان ایکس تست
model.evaluate(x_test , y_test , verbose = 2) 
print(".......................................................")
a = int(input("pleas enter sample of number : "))
#ری شیپ سمپل تست برای دادن به الگوریتم برای پیش بینی 
test_sample = x_test[a].reshape(1,28,28)
predict = model.predict(test_sample)
print(predict) #چون سافت مکس نزاشتیم اعداد منفی میبینیم
print(".......................................................")
#برطف کردن مشکل بالا برای احتمالات
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
print("probability :" , probability_model.predict(test_sample))
print(".......................................................")
#دسدن دقت تشخیص سمپل
print("sampel accuaracy :" , probability_model.predict(test_sample).max())
print(".......................................................")
#دیدن مدل برنده
print("sampel won :" , probability_model.predict(test_sample).argmax())
print(".......................................................")
#دیدن سمپل
plt.imshow(test_sample[0] , cmap= 'gray')
plt.title(y_test[a])
plt.show()




