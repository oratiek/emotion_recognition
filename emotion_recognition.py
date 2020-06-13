# 画像を扱うためのパッケージをインポート
import cv2
# 画像を配列に変更するパッケージをkerasからインポート
from keras.preprocessing.image import img_to_array
# 学習モデルを読み込むパッケージをkerasからインポート
from keras.models import load_model
# 配列を扱うためのパッケージをインポート
import numpy as np

# 学習モデルで表情を分類する関数を定義
def classify(img):
        # 画像のサイズを48x48に変更する
	img = cv2.resize(img,(48,48))
        # 画像のデータ型を浮動小数点に変更して255で割って正規化する
	img = img.astype("float32")/225.0
        # 画像を配列に変更する
	img = img_to_array(img)
        # 配列の形を学習モデルで分析できる形に変更する
	img = np.expand_dims(img, axis=0)
        # 学習モデルに通して結果をresultに代入する
	result = model.predict(img)[0]
        # resultの中にはそれぞれの表情の確率のデータが入っているのでその中で最も高い数値のインデックスと、元データを戻り値にする
	return result.argmax(), result

# モデルを読み込む
model = load_model("model.h5")
# OpenCVで表示する文字のフォントを指定する
fontType = cv2.FONT_HERSHEY_SIMPLEX
# インデックスで表情の文字データを取るための配列を定義する
EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]
# OpenCVで顔認識をするためのデータを読み込む
cascade = cv2.CascadeClassifier("cascade.xml")
# ウェブカメラからのキャプチャを取得する
cap = cv2.VideoCapture(0)
# 映像は画像の連番なので、映像で処理するためにループ処理を組む
while True:
        # キャプチャ(cap)から取得できているかのブール値(ret)と画像データ(frame)を取得する
	ret, frame = cap.read()
        # frameの向きを左右反転させて鏡と同じ向きにする
	frame = cv2.flip(frame, 1)
        # 画像を白黒に変更する（カラーよりもデータ量が減り認識し易くなるため）
	img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # 白黒にしたキャプチャから顔を認識してその座標（x,y,w,h）をfacesに代入する
	faces = cascade.detectMultiScale(img, 1.3, 5)
        # facesの中から座標データを取り出す
	for x,y,w,h in faces:
                # 顔の部分だけを取り出す
		face = img[y:y+h,x:x+w]
                # 始めに定義した表情を分類する関数classifyを呼び出す
		emotion, prob = classify(face)
                # どの表情が出たかを出力
		print(EMOTIONS[emotion])
                # OpenCVを使って結果を描画する
		cv2.putText(frame,EMOTIONS[emotion],(x,y), fontType, 3,(255,0,0),2,cv2.LINE_AA)
		cv2.putText(frame,"Angry:"+str(prob[0]),(0,100), fontType, 1,(255,0,0),2,cv2.LINE_AA)
		cv2.putText(frame,"Disgust:"+str(prob[1]),(0,200), fontType, 1,(255,0,0),2,cv2.LINE_AA)
		cv2.putText(frame,"Scared:"+str(prob[2]),(0,300), fontType, 1,(255,0,0),2,cv2.LINE_AA)
		cv2.putText(frame,"Happy:"+str(prob[3]),(0,400), fontType, 1,(255,0,0),2,cv2.LINE_AA)
		cv2.putText(frame,"Sad:"+str(prob[4]),(0,500), fontType, 1,(255,0,0),2,cv2.LINE_AA)
		cv2.putText(frame,"Surprised:"+str(prob[5]),(0,600), fontType, 1,(255,0,0),2,cv2.LINE_AA)
		cv2.putText(frame,"Neutral:"+str(prob[6]),(0,700), fontType, 1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),10)
        # 処理を行ったframeを表示する
	cv2.imshow("frame",frame)
        # キー入力を待って、「q」が押された時にbreakしてループを抜ける(ウィンドウのばつで消しても処理は終わらない、もしそうなって処理が止まらなくなったらターミナル上でctl+Cで強制終了させる)
	if cv2.waitKey(1) == ord("q"):
		break
