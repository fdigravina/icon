import pandas as pd
import numpy as np
import random

import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

def converti (situazione):
    if situazione == 0:
        return 'sole'
    if situazione == 1:
        return 'pioggia_lieve'
    if situazione == 2:
        return 'pioggia_media'
    return 'pioggia_forte'

df = pd.read_csv ('progetto finale/comuni_completo.csv')

nomiS = df['nome'].to_numpy()
latS = df['lat'].to_numpy(dtype=float)
lonS = df['lon'].to_numpy(dtype=float)

'''
k = int(len(nomi)*0.05)
idxs = random.sample(population=range(0, len(nomi)), k=k)

nomi = [nomi[idx] for idx in idxs]
lat = [lat[idx] for idx in idxs]
lon = [lon[idx] for idx in idxs]
'''

cities = ['Milano', 'Roma', 'Napoli', 'Torino', 'Genova', 'Bologna', 'Firenze', 'Bari', 'Catania', 'Venezia', 'Verona', 'Padova',
          'Brescia', 'Taranto', 'Prato', 'Modena', 'Reggio Calabria', 'Reggio Emilia', 'Perugia', 'Livorno', 'Ravenna',
          'Cagliari', 'Foggia', 'Rimini', 'Salerno', 'Ferrara', 'Latina', 'Giugliano in Campania', 'Monza', 'Siracusa', 'Pescara',
          'Bergamo', 'Forlì', 'Trento', 'Vicenza', 'Terni', 'Bolzano', 'Novara', 'Piacenza', 'Ancona', 'Andria', 'Udine', 'Arezzo',
          'Cesena', 'Lecce', 'La Spezia', 'Pesaro', 'Alessandria', 'Barletta', 'Catanzaro', 'Potenza', 'Massa', 'Carrara', 'Lucca',
          'Brindisi', 'Como', 'Grosseto', 'Sesto San Giovanni', 'Pozzuoli', 'Varese', 'Asti', 'Casoria', 'Cinisello Balsamo', 'Caserta',
          'Guidonia Montecelio', 'Castellammare di Stabia', 'L Aquila', 'Ragusa', 'Quartu Sant Elena', 'Agrigento', 'Cremona', 'Carpi',
          'Foligno', 'Tivoli', 'Pistoia', 'Imola', 'Altamura', 'Trapani', 'Cosenza', 'Viterbo', 'Aprilia', 'Savona', 'Benevento', 'Crotone',
          'Marano di Napoli', 'Matera', 'Acerra', 'Campobasso', 'Faenza', 'Cuneo', 'Cava de Tirreni', 'Lodi', 'Vercelli', 'Teramo',
          'Castrovillari', 'Pomezia', 'Scafati', 'San Severo', 'Vittoria', 'Gallarate', 'Saronno', 'Chioggia', 'Sanremo', 'Manfredonia',
          'Molfetta', 'Afragola', 'Scandicci', 'Busto Arsizio', 'Rho', 'Cervia', 'Avellino', 'Rieti', 'Aversa', 'Fano', 'Siena', 'Pordenone',
          'Jesi', 'Bisceglie', 'Ercolano', 'Acireale', 'San Giovanni Rotondo', 'Nichelino', 'Pompei', 'Caltanissetta', 'Montesilvano',
          'Cavaion Veronese', 'Casalnuovo di Napoli', 'San Donà di Piave', 'San Giuliano Milanese', 'San Giovanni la Punta', 'Sora',
          'Civitavecchia', 'Canosa di Puglia', 'Brugherio', 'Cernusco sul Naviglio', 'San Giovanni in Persiceto', 'San Miniato', 'Seregno',
          'Mondragone', 'Sant Antimo', 'Volla', 'Osimo', 'Suzzara', 'Nocera Superiore', 'San Felice a Cancello', 'Sant Anastasia', 'Orvieto',
          'Cassino', 'Ascoli Piceno', 'Acquaviva delle Fonti', 'Arzignano', 'Corato', 'Casale Monferrato', 'Portici', 'Sarzana', 'Minturno',
          'Villafranca di Verona', 'Civitanova Marche', 'Cisterna di Latina', 'Fidenza', 'Eboli', 'Porto Torres', 'Sant Antonio Abate',
          'San Giovanni Valdarno', 'Sant Agata de Goti', 'Gela', 'Gorgonzola', 'Lecco', 'Schio', 'Tortona', 'Lainate', 'San Mauro Torinese',
          'Rovigo', 'Pinerolo', 'Gorizia', 'Gaeta', 'Ardea', 'Formia', 'Rosignano Marittimo', 'Seriate', 'Spoleto', 'Monterotondo', 'Nettuno',
          'Alba', 'Piombino', 'Impruneta', 'Trani', 'Orta Nova', 'Rende', 'Vittorio Veneto', 'Maddaloni', 'Erba', 'Santa Maria Capua Vetere',
          'Casoria', 'Caivano', 'Santeramo in Colle', 'Copertino', 'Gravina in Puglia', 'Caltagirone', 'Chiavari', 'Giarre', 'Portogruaro',
          'Sarno', 'Frosinone', 'Lomazzo', 'Montebelluna', 'Porto Empedocle', 'Conegliano', 'Melzo', 'Acquaviva Picena', 'Savigliano',
          'Assemini', 'Cerignola', 'Avezzano', 'Sulmona', 'Chiari', 'San Giovanni Teatino', 'Bagnoli Irpino', 'Alpignano', 'Ciriè',
          'Cassano Magnago', 'Vittuone', 'Lesmo', 'Limbiate', 'Lissone', 'Quarto', 'Scanzorosciate', 'Senago', 'Bresso', 'Cernobbio',
          'Novate Milanese', 'Cologno Monzese', 'Bovisio-Masciago', 'Baranzate', 'Paderno Dugnano', 'Corsico', 'Cinisello Balsamo',
          'Sesto San Giovanni', 'Rho', 'Vimercate', 'Seregno', 'Desio', 'Bollate', 'Legnano', 'Lainate', 'Arese', 'Caronno Pertusella',
          'Cesano Maderno', 'Meda', 'Garbagnate Milanese', 'Monza', 'Cormano', 'Bareggio', 'Cusano Milanino', 'Arluno', 'Muggiò', 'Solaro',
          'Busto Garolfo', 'Novate Milanese']

nomi = []
lat = []
lon = []

for idx in range(len(nomiS)):
    if nomiS[idx] in cities:
        if random.random() < 0.5:
            nomi.append(nomiS[idx])
            lat.append(latS[idx])
            lon.append(lonS[idx])

n = len(nomi)
dim = int(n * 0.9)

pioggia = random.choices(population=[0, 1, 2, 3], weights=[0.6, 0.2, 0.13, 0.07], k=dim)    # sole, pioggia lieve, media, forte
pioggia.extend([-1] * (n - dim))    # -1 indica che non conosco la situazione meteorologica
random.shuffle(pioggia)

distanze = np.zeros (n**2)
idx = 0

with open ('progetto finale/predici.pl', 'w') as file:
    
    for i in range (n):
        for j in range (n):
            if i == j:
                idx += 1
                continue
            distanze[idx] = haversine(lat[i], lon[i], lat[j], lon[j])
            line = 'arco(' + str(i) + ', ' + str(j) + ', ' + str(int(distanze[idx])) + ').\n'
            file.write(line)
            idx += 1
    
    for i in range (n):
        if pioggia[i] == -1:
            continue
        line = 'situazione(' + str(i) + ', ' + converti(pioggia[i]) + ').\n'
        file.write(line)

lim_velocita = 90   # strade extraurbane

tempi = np.zeros (n**2)

for i in range(n**2):
    tempi[i] = distanze[i] / lim_velocita  # tempo = spazio / velocità

idx = 0

with open ('progetto finale/nomi.csv', 'w', encoding='utf-8') as f:
    for i in range(n):
        stringa = str(i) + ',' + nomi[i] + '\n'
        f.write (stringa)

with open ('progetto finale/dataset.csv', 'w', encoding='utf-8') as f:
    for i in range (n):
        for j in range (n):
            stringa = nomi[i] + ',' + nomi[j] +',' + str(tempi[idx] * 60) + '\n'
            f.write(stringa)
            idx += 1