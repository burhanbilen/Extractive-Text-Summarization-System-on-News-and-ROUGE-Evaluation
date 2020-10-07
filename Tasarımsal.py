import re
import codecs
import numpy as np
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from tkinter import Button, Tk, Frame, Scrollbar, StringVar, Label, Text, VERTICAL, W, N, S, E, WORD, INSERT, END
from tkinter.filedialog import askopenfilename, asksaveasfile
from numpy import dot
from numpy.linalg import norm

class metin_ozeti:
    def __init__(self):
        self.pencere = Tk()

        self.pencere.title("Metin Özeti")
        self.pencere.geometry('800x520')
        self.pencere.resizable(0, 0)

        self.cerceve = Frame(self.pencere)
        self.cerceve.pack()

        self.cerceve.configure(background="#140718")
        self.pencere.configure(background="#140718")

        self.BtnDosyaAc = Button(self.cerceve, width=13, height=1, text='Metin Ekle', command=self.Ozet_Cıkar)
        self.BtnDosyaAc.grid(row=0, column=0, padx=0, pady=8, sticky=W)

        self.dosyayolu = StringVar()
        Label(self.cerceve, textvariable=self.dosyayolu, fg="white", bg="#140718").grid(row=0, column=0, sticky=E)
        Label(self.cerceve, text="Özet: ", fg="white", bg="#140718").grid(row=1, column=0, sticky=W)

        kaydırma_cubugu = Scrollbar(self.cerceve, orient=VERTICAL)
        kaydırma_cubugu.grid(row=2, column=1, padx=0, pady=0, sticky=N + S + W)

        self.yazı = Text(self.cerceve, width=90, height=24.5, wrap=WORD, bd=5, yscrollcommand=kaydırma_cubugu.set)
        self.yazı.grid(row=2, column=0, sticky=S + N + E + W)

        self.BtnOzetKaydet = Button(self.cerceve, width=30, height=1, text='Özeti Kaydet', command=self.Ozet_Kaydet)

        kaydırma_cubugu.config(command=self.yazı.yview)

        self.pencere.mainloop()

    def BOW_vektormetin(self, metin, cumle):
        vocabulary = set(list(word_tokenize(metin)))
        d = dict()
        for i in vocabulary:
            d[i] = 0

        for i in word_tokenize(cumle):
            if i in vocabulary:
                d[i] = d.get(i, 0) + 1

        matris = list(d.values())
        norm = np.linalg.norm(matris)
        normal_array = matris / norm
        return normal_array

    def Ozet_Cıkar(self):
        metinyolu = askopenfilename(filetypes=[("Metin Belgesi", "*.txt")], title="Bir Metin Seç")
        if (metinyolu[::-1].split('.'))[0] == "txt":
            metin_veri = codecs.open(metinyolu, 'r',encoding='unicode_escape').read()
            if metin_veri is None or metin_veri == "":
                print("Dosya içinde veri bulunamadı!")
                return

            if metin_veri != None or metin_veri != "":
                self.BtnOzetKaydet.grid(row=3, column=0, padx=5, pady=5)
        else:
            print("Dosya Hatası!")
            return

        self.dosyayolu.set(metinyolu)

        stopWords = list(stopwords.words('english'))  
        kelimeler = word_tokenize(str(metin_veri))
        cumleler = sent_tokenize(str(metin_veri))

        ayrıstırma = pos_tag(kelimeler)
        ozel_isimler = [kelime for kelime, pos in ayrıstırma if pos == 'NNP']

        stemmer = PorterStemmer()
        kelime_sıklık_tablosu = dict()

        for kelime in kelimeler:
            kelime = stemmer.stem(kelime)
            if kelime in stopWords:
                continue
            if kelime in kelime_sıklık_tablosu:
                kelime_sıklık_tablosu[kelime] += 1
            else:
                kelime_sıklık_tablosu[kelime] = 1

        en_sık_kelime = max(kelime_sıklık_tablosu, key=kelime_sıklık_tablosu.get)
        en_buyuk_sayı = int(kelime_sıklık_tablosu[en_sık_kelime])

        def TF(aranan, metin):
            i = 0
            kelimeler = word_tokenize(metin)
            for x in kelimeler:
                if aranan == x:
                    i += 1
            tf_sonuc = i / len(kelimeler)
            i = 0
            return tf_sonuc

        kelime_agırlık_tablosu = dict()
        for anahtar, deger in kelime_sıklık_tablosu.items():
            if anahtar == "." or anahtar == "," or anahtar == "%" or anahtar == "!" or anahtar == "'" or anahtar == "?" or anahtar == "...":
                continue
            kelime_agırlık_tablosu[anahtar] = deger / en_buyuk_sayı
        en_cok_puanlanan_kelime = max(kelime_agırlık_tablosu, key=kelime_agırlık_tablosu.get)

        kelime_sayacı_temp = 0
        kelime_sayacı = 0
        for cumle in cumleler:
            for i in word_tokenize(cumle):
                kelime_sayacı_temp += 1
            if kelime_sayacı_temp >= kelime_sayacı:
                kelime_sayacı = kelime_sayacı_temp

        regex = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'
        numerik_veri_sayaci = 0
        cumle_skor_tablosu = dict()
        ozel_isim_sozlugu = dict()
        ozel_isim_sayac = 0
        cumle_konum_sayacı = 0
        for cumle in cumleler:
            i = 0
            cumle_konum_sayacı += 1
            kelimeler = list(word_tokenize(cumle))
            stopwordsuz_kelime_miktarı = 0
            for anahtar in kelime_sıklık_tablosu:
                if anahtar in cumle:
                    stopwordsuz_kelime_miktarı += 1
                    if cumle[:20] in cumle_skor_tablosu:
                        cumle_skor_tablosu[cumle[:20]] += kelime_sıklık_tablosu[anahtar]
                    elif cumle[:20] not in cumle_skor_tablosu:
                        cumle_skor_tablosu[cumle[:20]] = kelime_sıklık_tablosu[anahtar]
                    if anahtar in cumleler[0]:
                        cumle_skor_tablosu[cumle[:20]] += 0.00005
                    cumle_skor_tablosu[cumle[:20]] += (TF(anahtar, metin_veri) / kelime_agırlık_tablosu[en_cok_puanlanan_kelime])
            cumle_skor_tablosu[cumle[:20]] += dot(self.BOW_vektormetin(metin_veri, cumle),self.BOW_vektormetin(metin_veri, cumleler[0])) / (norm(self.BOW_vektormetin(metin_veri, cumle)) * norm(self.BOW_vektormetin(metin_veri, cumleler[0])))
            cumle_skor_tablosu[cumle[:20]] += dot(self.BOW_vektormetin(metin_veri, cumle),self.BOW_vektormetin(metin_veri, cumleler[len(cumleler) - 1])) / (norm(self.BOW_vektormetin(metin_veri, cumle)) * norm(self.BOW_vektormetin(metin_veri, cumleler[len(cumleler) - 1])))

            if ozel_isimler[i] in list(word_tokenize(cumle)):
                ozel_isim_sayac += 1
                
            if re.search(regex,cumle) is not None:
                for j in re.finditer(regex, cumle):
                    numerik_veri_sayaci += 1
                    
            cumle_skor_tablosu[cumle[:20]] += numerik_veri_sayaci / len(cumle)
            cumle_skor_tablosu[cumle[:20]] += (len(cumleler) - cumle_konum_sayacı) / len(cumleler)
            cumle_skor_tablosu[cumle[:20]] += len(kelimeler) / kelime_sayacı

            if kelimeler[i] in ozel_isim_sozlugu.keys():
                cumle_skor_tablosu[cumle[:20]] += ozel_isim_sayac / len(cumle)

            cumle_skor_tablosu[cumle[:20]] = cumle_skor_tablosu[cumle[:20]] / stopwordsuz_kelime_miktarı
            i += 1
            
        skor = 0
        for cumle in cumle_skor_tablosu:
            skor += cumle_skor_tablosu[cumle]

        ortalama_skor = (skor / len(cumle_skor_tablosu))

        cumle_konum_sayacı = 0
        self.ozet = ''
        for cumle in cumleler:
            if cumle[:20] in cumle_skor_tablosu and cumle_skor_tablosu[cumle[:20]] >= ortalama_skor:
                self.ozet += " " + cumle

        self.yazı.delete(1.0, END)
        self.yazı.insert(INSERT, self.ozet)

    def Ozet_Kaydet(self):
        dosya_adi = ""
        for i in self.dosyayolu.get()[::-1]:
            if i in "txt.":
                pass
            elif i == "/":
                break
            else:
                dosya_adi += i
        try:
            kayıt = asksaveasfile(initialfile=dosya_adi[::-1] + " - Özet", mode='w', title="Özeti Kaydet", filetypes=[("Metin Belgesi", ".txt")], defaultextension=".txt")
            kayıt.write(self.ozet)
            kayıt.close()
        except:
            raise ("Hata!")

metin_ozeti()
