import re
import nltk
import codecs
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.util import ngrams, skipgrams
from os import listdir
from os.path import isfile, join
import numpy as np
from numpy import dot
from numpy.linalg import norm
from itertools import combinations

class metin_ozeti:
    def __init__(self):
        self.surekli()

    def surekli(self):
        self.toplam_rouge_1 = 0
        self.toplam_rouge_2 = 0
        self.toplam_rouge_3 = 0
        self.toplam_rouge_4 = 0
        self.toplam_f_score_lcs = 0
        self.toplam_f_score_skip2 = 0

        klasor_konumu = "C:/Users/user/Desktop/Metin Özeti/BBC News Summary/News Articles/entertainment"
        dosyalar = [f for f in listdir(klasor_konumu) if isfile(join(klasor_konumu, f))]
        for i in dosyalar:
            self.Ozet_Cıkar(klasor_konumu + "/" + i)

        print("ROUGE-1: ", self.toplam_rouge_1 / len(dosyalar))
        print("ROUGE-2: ", self.toplam_rouge_2 / len(dosyalar))
        print("ROUGE-3: ", self.toplam_rouge_3 / len(dosyalar))
        print("ROUGE-4: ", self.toplam_rouge_4 / len(dosyalar))
        print("ROUGE-L: ", self.toplam_f_score_lcs / len(dosyalar))
        print("ROUGE-S: ", self.toplam_f_score_skip2 / len(dosyalar))

    def skip2gram(self, metin):
        merkez_kelime = ""
        skip2_liste = []
        kullanılan_merkez_kelime = []
        for i in range(len(metin) - 1):
            merkez_kelime = metin[i]
            kullanılan_merkez_kelime.append(merkez_kelime)
            for j in metin:
                if j not in kullanılan_merkez_kelime:
                    skip2_liste.append(merkez_kelime + " " + j)
        return skip2_liste

    def BOW_vektormetin(self, metin, cumle):  # Bag Of Word yöntemiyle cümleleri vektörleştirmek için kullandığımız metot
        vocabulary = set(list(word_tokenize(metin)))
        d = dict()
        for i in vocabulary:
            d[i] = 0

        for i in word_tokenize(cumle):
            if i in vocabulary:
                d[i] = d.get(i, 0) + 1

        matris = list(d.values())
        norm = np.linalg.norm(matris)
        normal_array = matris / norm  # Oluşan matristeki değerleri ölçeklendirme işlemi
        return normal_array

    def Ozet_Cıkar(self, dosya_yolu):
        metin_veri = codecs.open(dosya_yolu, 'r', encoding="utf-8").read()  # .py dosyamızın bulunduğu dizindeki .txt verilerini codecs kütüphanesi ile okuduk.
        if metin_veri is None or metin_veri == "":
            print("Dosya içinde veri bulunamadı!")
            return

        stopWords = list(stopwords.words('english'))  # Yaygın ve gereksiz olan sözcükleri, daha sonra kullanabilmek üzere bir listeye aktardık.

        kelimeler = word_tokenize(str(metin_veri))  # Okuduğumuz .txt belgesindeki tüm veriyi sadece kelimeler halinde elde ettik ve kelimeler adlı listeye aktardık.
        cumleler = sent_tokenize(str(metin_veri))  # Okuduğumuz .txt belgesindeki tüm veriyi sadece cümleler olarak elde ettik ve cumleler adlı listeye aktardık.

        ayrıstırma = pos_tag(kelimeler)  # Kelime ve etiket ayrışımı için kelime ve etiket tanımlaması yapıldı.
        ozel_isimler = [kelime for kelime, pos in ayrıstırma if pos == 'NNP']  # List comprehension yöntemiyle, mzel isim olan kelimeleri listeye ekledik.

        stemmer = PorterStemmer()  # Stemming için kök ayırıcı fonksiyon.
        kelime_sıklık_tablosu = dict()

        for kelime in kelimeler:
            kelime = stemmer.stem(kelime)  # Kelimeleri köklerine ayırdık.
            if kelime in stopWords:
                continue  # Eğer elime bir Stopword ise, es geçiyoruz.
            if kelime in kelime_sıklık_tablosu:
                kelime_sıklık_tablosu[kelime] += 1  # Sıklık sözlüğüne kelimeler ve adetleri eklendi.
            else:
                kelime_sıklık_tablosu[kelime] = 1

        en_sık_kelime = max(kelime_sıklık_tablosu, key=kelime_sıklık_tablosu.get)  # En sık kelime bulundu.
        en_buyuk_sayı = int(kelime_sıklık_tablosu[en_sık_kelime])  # En sık kelimenin kaç defa tekrar ettiği elde edildi.

        def TF(aranan, metin):
            i = 0
            kelimeler = word_tokenize(metin)
            for x in kelimeler:
                if aranan == x:
                    i += 1
            tf_sonuc = i / len(kelimeler)  # TF(TERM FREQUENCY) FORMÜLÜ
            i = 0
            return tf_sonuc

        kelime_agırlık_tablosu = dict()
        for anahtar, deger in kelime_sıklık_tablosu.items():
            if anahtar == "." or anahtar == "," or anahtar == "%" or anahtar == "!" or anahtar == "'" or anahtar == "?" or anahtar == "...":
                continue
            kelime_agırlık_tablosu[anahtar] = deger / en_buyuk_sayı  # TF- TERM FREQUENCY methodu ile ağırlık hesaplaması.
        en_cok_puanlanan_kelime = max(kelime_agırlık_tablosu, key=kelime_agırlık_tablosu.get)

        """en uzun cümleyi bulmak için kullandığım yol"""
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
            kelimeler = list(word_tokenize(cumle))  # Cümle değişkenine atanan cümlenin içindeki kelime miktarını buldu
            stopwordsuz_kelime_miktarı = 0  # Stopwordsuz kelime sayısını bulmak için oluşturuldu
            for anahtar in kelime_sıklık_tablosu:  # Kelime sıklık tablosunda, anahtar adlı değişken ile döngü kuruldu
                if anahtar in cumle:  # Sıklık tablosundan indisine uygun olan veriye eşitlenen anahtarın seçilen cümlede olup olmadığı kontrol edildi
                    stopwordsuz_kelime_miktarı += 1  # Bu arada stopwordsuz kelimeler, her cümle için sayıldı
                    if cumle[
                       :20] in cumle_skor_tablosu:  # Uzun ve kısa cümleler arası eşit puanlama yapılması için 0'dan 20'ye kadar olan karakterleri seçtik.
                        cumle_skor_tablosu[cumle[:20]] += kelime_sıklık_tablosu[anahtar]
                    elif cumle[:20] not in cumle_skor_tablosu:
                        cumle_skor_tablosu[cumle[:20]] = kelime_sıklık_tablosu[anahtar]
                    if anahtar in cumleler[0]:  # Cümlemizdeki kelimelerin,metindeki ilk cümlede olup olmadığını kontrol edip puanlama yaptık.
                        cumle_skor_tablosu[cumle[:20]] += 0.00005  # keyfi puanlama.
                    cumle_skor_tablosu[cumle[:20]] += (TF(anahtar, metin_veri) / kelime_agırlık_tablosu[
                        en_cok_puanlanan_kelime])  # Terim Ağırlıklandırma formülü (tf kullanarak)
            """Mevcut cümlenin, metnin 'İLK' cümlesine benzerliğini bulmak için kosinüs benzerlik förmülüyle ağırlıklandırma işlemi"""
            cumle_skor_tablosu[cumle[:20]] += dot(self.BOW_vektormetin(metin_veri, cumle),self.BOW_vektormetin(metin_veri, cumleler[0])) / (norm(self.BOW_vektormetin(metin_veri, cumle)) * norm(self.BOW_vektormetin(metin_veri, cumleler[0])))
            """Mevcut cümlenin, metnin 'SON' cümlesine benzerliğini bulmak için kosinüs benzerlik förmülüyle ağırlıklandırma işlemi"""
            cumle_skor_tablosu[cumle[:20]] += dot(self.BOW_vektormetin(metin_veri, cumle),self.BOW_vektormetin(metin_veri, cumleler[len(cumleler) - 1])) / (norm(self.BOW_vektormetin(metin_veri, cumle)) * norm(self.BOW_vektormetin(metin_veri, cumleler[len(cumleler) - 1])))

            if ozel_isimler[i] in cumle:
                ozel_isim_sayac += 1  # Mevcut cümle içerisindeki özel isim adedi sayıldı.
            if re.search(regex,cumle) is not None:  # Sayısal verileri, regular expression ile, bulabilmek için kullanılan sorgu ve döngü grubu
                for j in re.finditer(regex, cumle):
                    numerik_veri_sayaci += 1
            cumle_skor_tablosu[cumle[:20]] += numerik_veri_sayaci / len(cumle)  # Nümerik veri hesaplaması sonucunda kullanılan ağırlıklandırma formülü
            cumle_skor_tablosu[cumle[:20]] += (len(cumleler) - cumle_konum_sayacı) / len(cumleler)  # cümle konumuna göre puanlandırma
            cumle_skor_tablosu[cumle[:20]] += len(kelimeler) / kelime_sayacı  # uzun cümlelerin puanlaması

            if kelimeler[i] in ozel_isim_sozlugu.keys():  # Özel isim veya kelime kontrolü ve puanlaması yapıldı.
                cumle_skor_tablosu[cumle[:20]] += ozel_isim_sayac / len(cumle)

            cumle_skor_tablosu[cumle[:20]] = cumle_skor_tablosu[cumle[:20]] / stopwordsuz_kelime_miktarı  # Cümlelere ait puanlama yöntemi.
            i += 1
        skor = 0  # Genel skor değişeni tanımlandı.
        for cumle in cumle_skor_tablosu:  # Sözlüğümüzdeki cümleleri tek tek işliyoruz.
            skor += cumle_skor_tablosu[cumle]  # Herbir cümlenin skorunu toplayıp toplam skoru elde ediyoruz.

        ortalama_skor = (skor / len(cumle_skor_tablosu))  # Tüm skoru, skor sözlüğü uzunluğuna bölerek elde ediyoruz.

        cumle_konum_sayacı = 0
        self.ozet = ''
        for cumle in cumleler:
            if cumle[:20] in cumle_skor_tablosu and cumle_skor_tablosu[
                cumle[:20]] >= ortalama_skor:  # Cumle varlık kontrolü ve skor kıyaslaması yapıldı.
                self.ozet += " " + cumle  # Cümleler, özet değişkeninde toplandı.

        """özet yazı kısmı"""
        ozetteki_cumleler = list(
            sent_tokenize(str(self.ozet)))  # metric hesaplamaları için sistem özetindeki cümleleri elde etme

        """referans yazı kısmı"""  # Bu kısım, elde edilen yazının ideal yanı referans olan özete erişmek için dosya yolu elde edeceğimiz kısım
        taksim_sayac = 0
        idealozet_konum = ""
        yol = ""
        yol2 = ""
        for i in reversed(dosya_yolu):
            yol += i
            if i == "/":
                taksim_sayac += 1
                if taksim_sayac == 2:
                    idealozet_konum = "Summaries" + yol[::-1]
            if taksim_sayac >= 3:
                yol2 += i
        idealozet_konum = yol2[::-1] + idealozet_konum

        self.referans_veri = codecs.open(Path(idealozet_konum), 'r', encoding='utf-8').read()  # referans(ideal) özeti okuma.

        referansverideki_cumleler = list(self.idealozetteki_cumleler(self.referans_veri))  # ideal özetteki cümle sayısını nltk'nın sent_tokenize() metoduyla yaptığımda yanlış sonuç verdiğinden regular expression kullandım.

        formul_sayac = 0  # aşağıda kullanılan precision ve recall formülleri için, eşleşen cümle sayısını tutan bir değişken oluşturduk.
        for i in ozetteki_cumleler:
            if i in referansverideki_cumleler:
                formul_sayac += 1

        precision = (formul_sayac) / len(ozetteki_cumleler)  # precision, recall ve f-score hesaplamaları
        recall = (formul_sayac) / len(referansverideki_cumleler)
        f_score = (2 * precision * recall) / (precision + recall + 10 ** (-30))  # paydaya 10 üzeri eksi 30gibi küçük bir sayı eklememizin sebebi paydanın bazen 0 çıkıp programın durmasıdır.

        str_a = self.referans_veri
        str_b = self.ozet

        def lcs(s1, s2):  # en uzun alt stringi hesaplamak için bir kod.
            matrix = [["" for x in range(len(s2))] for x in range(len(s1))]
            for i in range(len(s1)):
                for j in range(len(s2)):
                    if s1[i] == s2[j]:
                        if i == 0 or j == 0:
                            matrix[i][j] = s1[i]
                        else:
                            matrix[i][j] = matrix[i - 1][j - 1] + s1[i]
                    else:
                        matrix[i][j] = max(matrix[i - 1][j], matrix[i][j - 1], key=len)
            cs = matrix[-1][-1]
            return len(cs)

        precision_lcs = lcs(str_a, str_b) / len(self.ozet)  # Rouge-L için precision-lcs, recall-lcs ve f-score-lcs hesaplaması
        recall_lcs = lcs(str_a, str_b) / len(self.referans_veri)  # precision değerini, en uzun alt string uzunluğunu özetin uzunluğuna ve recall içinse, aynı işlemi, referans özet uzunluğuna bölerek elde ediyoruz.

        beta2 = (precision_lcs / recall_lcs) ** 2  # f-score hesabı için gerekli olan beta^2 sayısı
        f_score_lcs = ((1 + beta2) * recall_lcs * precision_lcs) / (recall_lcs + (precision_lcs * beta2))

        """N-GRAM HESAPLAMALARI (ROUGE-N İÇİN)"""

        def extract_ngrams(data,num):  # rouge-n testi için ngram algoritması
            n_grams = ngrams(nltk.word_tokenize(data), num)
            return [' '.join(grams) for grams in n_grams]

        ozet_unigram = extract_ngrams(self.ozet, 1)  # kodun oluşturduğu özet için unigram hesabı
        ozet_bigram = extract_ngrams(self.ozet, 2)  # kodun oluşturduğu özet için bigra   m hesabı
        ozet_trigram = extract_ngrams(self.ozet, 3)  # kodun oluşturduğu özet için trigram hesabı
        ozet_4gram = extract_ngrams(self.ozet, 4)  # kodun oluşturduğu özet için fourgram hesabı

        ideal_unigram = extract_ngrams(self.referans_veri, 1)  # ideal özet için unigram hesabı
        ideal_bigram = extract_ngrams(self.referans_veri, 2)  # ideal özet için bigram hesabı
        ideal_trigram = extract_ngrams(self.referans_veri, 3)  # ideal özet için trigram hesabı
        ideal_4gram = extract_ngrams(self.referans_veri, 4)  # ideal özet için fourgram hesabı

        list_uni = list()
        for e in ozet_unigram:  # sistem özeti ve ideal özette bulunan unigram eşleşmesi sayısı
            if e in ideal_unigram:
                list_uni.append(e)

        list_bi = list()
        for e in ozet_bigram:  # sistem özeti ve ideal özette bulunan bigram eşleşmesi sayısı
            if e in ideal_bigram:
                list_bi.append(e)

        list_tri = list()
        for e in ozet_trigram:  # sistem özeti ve ideal özette bulunan trigram eşleşmesi sayısı
            if e in ideal_trigram:
                list_tri.append(e)

        list_4 = list()
        for e in ozet_4gram:  # sistem özeti ve ideal özette bulunan fourgram eşleşmesi sayısı
            if e in ideal_4gram:
                list_4.append(e)

        rouge_1 = len(list_uni) / len(ideal_unigram)
        rouge_2 = len(list_bi) / len(ideal_bigram)
        rouge_3 = len(list_tri) / len(ideal_trigram)
        rouge_4 = len(list_4) / len(ideal_4gram)
        # rouge-1 rouge-2 ve rouge-3 hesaplamaları

        """------------------------------------------------------------------------------------------"""

        def kombinasyon(n, r):  # ROUGE-S formülü için kombinasyon hesaplayan metot
            C = n  # C: kombinasyon
            for i in range(1, r):
                n -= 1
                C *= n
            for i in range(2, r + 1):
                C = C / i
            if C == 0:
                return 1
            else:
                return C

        def rouge_s_bulucu(sistem_ozeti_bigram, ideal_ozet_bigram):
            skip2_sayac = 0
            for i in sistem_ozeti_bigram:
                for j in ideal_ozet_bigram:
                    if i == j:
                        skip2_sayac += 1
            return skip2_sayac

        x = self.referans_veri
        y = self.ozet

        referans_veri_kelimeler = list(word_tokenize(self.referans_veri))
        ozet_kelimeler = word_tokenize(self.ozet)

        k1 = list(skipgrams(word_tokenize(x), 2, 2))
        k2 = list(skipgrams(word_tokenize(y), 2, 2))

        m = len(list(word_tokenize(self.ozet)))
        n = len(list(word_tokenize(self.referans_veri)))

        precision_skip2 = (rouge_s_bulucu(k1, k2) / len(k1)) + 10**(-30)
        recall_skip2 = (rouge_s_bulucu(k1, k2) / len(k2)) + 10**(-30) #paydanın 0 olmasını önlemek için küçük bir değer ekledik.

        beta2_s2 = (precision_skip2 / recall_skip2) ** 2
        f_score_skip2 = ((1 + beta2_s2) * precision_skip2 * recall_skip2) / (recall_skip2 + (beta2_s2 * precision_skip2))
        
        self.toplam_rouge_1 += rouge_1
        self.toplam_rouge_2 += rouge_2
        self.toplam_rouge_3 += rouge_3
        self.toplam_rouge_4 += rouge_4
        self.toplam_f_score_lcs += f_score_lcs
        self.toplam_f_score_skip2 += f_score_skip2

    def idealozetteki_cumleler(self, text):  # regular expression yöntemiyle yazıyı cümlelerine ayırmak için.
        self.alphabets = "([A-Za-z])"
        self.prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
        self.suffixes = "(Inc|Ltd|Jr|Sr|Co)"
        self.starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
        self.acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
        self.websites = "[.](com|net|org|io|gov)"
        self.digits = "([0-9])"
        text = " " + text + "  "
        text = text.replace("\n", " ")
        text = re.sub(self.prefixes, "\\1<prd>", text)
        text = re.sub(self.websites, "<prd>\\1", text)
        text = re.sub(self.digits + "[.]" + self.digits, "\\1<prd>\\2", text)
        if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
        text = re.sub("\s" + self.alphabets + "[.] ", " \\1<prd> ", text)
        text = re.sub(self.acronyms + " " + self.starters, "\\1<stop> \\2", text)
        text = re.sub(self.alphabets + "[.]" + self.alphabets + "[.]" + self.alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>", text)
        text = re.sub(self.alphabets + "[.]" + self.alphabets + "[.]", "\\1<prd>\\2<prd>", text)
        text = re.sub(" " + self.suffixes + "[.] " + self.starters, " \\1<stop> \\2", text)
        text = re.sub(" " + self.suffixes + "[.]", " \\1<prd>", text)
        text = re.sub(" " + self.alphabets + "[.]", " \\1<prd>", text)
        if "”" in text: text = text.replace(".”", "”.")
        if "\"" in text: text = text.replace(".\"", "\".")
        if "!" in text: text = text.replace("!\"", "\"!")
        if "?" in text: text = text.replace("?\"", "\"?")
        text = text.replace(".", ".<stop>")
        text = text.replace("?", "?<stop>")
        text = text.replace("!", "!<stop>")
        text = text.replace("<prd>", ".")
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]
        return sentences

metin_ozeti()
