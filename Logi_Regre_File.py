import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

# ML lab 04-2: TensorFlow로 파일에서 데이타 읽어오기 4분 위치
xy = np.loadtxt ('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
# 앞의 : Colon 은 모든 행에서, 콤마 후 열은 0번 열에서 마지만 열 제외하고...
x_data = xy[:,0:-1]
# 모든 Row 행에서 , 마지막 값만 가져와라
y_data = xy[:,[-1]]

X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None,1])
W = tf.Variable(tf.random_normal([8,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X,W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Cast Function 은 next equation 의 condition 에 따라 True or False cast
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# Predicted 와 실 Y 값이 Equal 되서 정확도가 높았나 평균을 내어보자
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
# 이제 학습 시작
# 세션을 하나 정의
with tf.Session() as sess:
    # 세션에서 사용할 variable 을 초기화
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 ==0:
            print (step, cost_val)

    # 이제 학습 결과 출력
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict= {X: x_data, Y: y_data})
    print("\nHypothesis : ", h, "\nResult (Y) : ", c, "\n Accuracy : ", a)
