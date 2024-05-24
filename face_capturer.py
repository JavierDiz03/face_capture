import tkinter as tk
from tkinter import messagebox
import cv2
import os
import imutils
import time
import subprocess

dataPath = "C:/Users/javie/Desktop/cosas/ReconocimientoFacial/Data"

def login():
    global personName
    personName = entry_username.get()
    if personName.strip() == "":
        messagebox.showerror("Error", "Please, type your user.")
    else:
        personPath = os.path.join(dataPath, personName)
        if os.path.exists(personPath):
            root.destroy()
            subprocess.run(["python", "saved_faces.py"])
            cv2.destroyWindow('frame')
        else:
            messagebox.showinfo("Unknown user", "Press enter to auto-authenticate")
            register_new_user(personName)

def register_new_user(personName):
    personPath = os.path.join(dataPath, personName)
    os.makedirs(personPath)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()

        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rostro = auxFrame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(personPath, 'rostro_{}.jpg'.format(count)), rostro)
            count += 1
            if count == 300:
                print("Scan complete")
                os.system('python IA_trainer.py')
                time.sleep(1)
                cap.release()
                cv2.destroyWindow('frame')
                break

        cv2.imshow('frame', frame)

        k = cv2.waitKey(1)
        if k == 27 or count >= 300:
            break

    cap.release()
    cv2.destroyAllWindows()

root = tk.Tk()
root.title("Login and Register")
root.geometry("400x300")
root.configure(bg='#0A0F1F')

# Configurar los elementos de la ventana
label_username = tk.Label(root, text="User:", bg='#0A0F1F', fg='white', font=('Helvetica', 12))
label_username.pack(pady=(30, 10))

entry_username = tk.Entry(root, font=('Helvetica', 12))
entry_username.pack(pady=(0, 20))

btn_login = tk.Button(root, text="Login", command=login, font=('Helvetica', 12), bg='#29a617', fg='white')
btn_login.pack()

root.mainloop()
