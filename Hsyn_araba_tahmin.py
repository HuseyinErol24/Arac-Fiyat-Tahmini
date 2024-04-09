from flask import Flask, render_template, request
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = Flask(__name__)
model = load_model('tamam.keras')
scaler = MinMaxScaler()

@app.route('/')
def index():
    return render_template('araba_fiyat.html')


@app.route('/', methods=['POST'])
def tahmin():
    yil = float(request.form["Yıl"])
    sanziman = float(request.form["Şanzıman"])
    kilometre = float(request.form["Kilometre"])
    vergi = float(request.form["Vergi"])
    mpg = float(request.form["Mil Başına Galon"])
    motor_hacmi = float(request.form["Motor Hacmi"])
    veri = np.array([[yil, sanziman, kilometre, vergi, mpg, motor_hacmi]])
    veri_normalized = scaler.transform(veri)
    tahmin = model.predict(veri_normalized)
    return render_template('araba_fiyat.html', tahmin=tahmin[0][0])


if __name__ == '__main__':
    app.run(debug=True)
