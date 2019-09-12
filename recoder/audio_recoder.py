"""
Pour la capture et de sons avec l'interface graphique utilisateur
Les données sont en suite envoyé dans le model pour la prediction
"""

from tkinter import *
import sounddevice as sd
import librosa
from scipy.io.wavfile import write
import numpy as np
from dataset.prediction import predict
raw_path = '../dataset/ESC-SEC/test/'

fs = 22050  # Sample rate
seconds = 4  # Duration of recording
n_sample = fs*seconds
sample = np.zeros((1, n_sample))
cancel = True
click = 1


def debit_capture():
    global sample
    global cancel
    global click
    while True:
        lb_pred.config(text=listen_text)
        # Capture de son
        sample = sd.rec(int(n_sample), samplerate=fs, channels=1)
        sd.wait()  # On attend que la capture se termine
        # Faire la prediction du son
        lb_pred.config(text=understand_text)
        predected = predict(sample)
        show_voyant(predected)
        if cancel:
            lb_pred.config(text=predected)
            break


def enregistrer():
    write('output.wav', fs, sample)  # Save as WAV file


def stop_capture():
    global cancel
    cancel = True
    global lb_pred
    lb_pred.config(text=default_text)
    hide_voyant()


def show_voyant(classe):
    rouges = ["coup_feu", "feu", "sirene"]
    oranges = ["pas", "pluie", "aboiement"]
    verts = ["klaxon", "cognements"]
    global lb_rouge
    global lb_orange
    global lb_vert
    hide_voyant()

    if classe in rouges:
        lb_rouge.pack(side=TOP, pady=(30, 0))
    elif classe in verts:
        lb_vert.pack(side=BOTTOM, pady=(30, 0))
    elif classe in oranges:
        lb_orange.pack(side=BOTTOM, pady=(30, 0))


def hide_voyant():
    global lb_rouge
    global lb_orange
    global lb_vert

    lb_rouge.pack_forget()
    lb_orange.pack_forget()
    lb_vert.pack_forget()


widow = Tk()

# Personnaliser la fenetre
widow.title("SAAS")
widow.geometry("560x400")
widow.minsize(560, 400)
widow.iconbitmap("images/oreille.ico")
widow.config(background='#ff9c84')

# Le titre de l'application
lb_title = Label(widow, text="Système d'audio surveillance",
                 font=("Courrier", 20), bg='#ff9c84', fg='#000080')
lb_title.pack(side=TOP)
frame_cent = frame_voyant = Frame(widow, bg='#ff9c84')
frame_cent.pack(expand=YES)

# Voyant de groupe de son
frame_voyant = Frame(frame_cent, bg='#ff9c84')
img_rouge = PhotoImage(file='images/red.png')
img_vert = PhotoImage(file='images/vert.png')
img_orange = PhotoImage(file='images/orange.png')
lb_rouge = Label(frame_voyant, image=img_rouge, bg='#ff9c84')
lb_vert = Label(frame_voyant, image=img_vert, bg='#ff9c84')
lb_orange = Label(frame_voyant, image=img_orange, bg='#ff9c84')
lb_rouge.pack(side=TOP, pady=(30, 0))
lb_orange.pack(side=BOTTOM, pady=(30, 0))
lb_vert.pack(side=BOTTOM, pady=(30, 0))
hide_voyant()

frame_voyant.pack(side=RIGHT)

# La prédiction
default_text = '------||||-------||||-------'
listen_text = "Ecoute ..."
understand_text = "Compréhension ..."
lb_pred = Label(frame_cent, text=default_text, font=("Courrier", 30),
                 bg='#f2f2f2', fg='#dfad00', height=4)
lb_pred.pack(side=LEFT, padx=(0, 30))

# Bouton de controle
frame_buttom = Frame(widow, bg='#ff9c84')
btn_start = Button(frame_buttom, text='debut', font=("Arial", 14), bg='#801a00', fg='white', command=debit_capture)
btn_stop = Button(frame_buttom, text='Stoper', font=("Arial", 14), bg='#801a00', fg='white', command=stop_capture)
btn_enrg = Button(frame_buttom, text='Enregistrer', font=("Arial", 14), bg='#801a00', fg='white', command=enregistrer)
btn_start.pack(side=LEFT, padx=(10, 0))
btn_stop.pack(side=LEFT, padx=(10, 0))
btn_enrg.pack(side=RIGHT, padx=(10, 0))
frame_buttom.pack(side=BOTTOM, expand=YES)


# Afficher
widow.mainloop()
