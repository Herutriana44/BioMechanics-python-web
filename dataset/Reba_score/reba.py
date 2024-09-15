import pandas as pd

# Membaca tabel REBA dari file CSV
reba_table_a = pd.read_csv('dataset/Reba_score/TableA.csv')
reba_table_b = pd.read_csv('dataset/Reba_score/TableB.csv')
reba_table_c = pd.read_csv('dataset/Reba_score/TableC.csv')

def calculate_reba(trunk_pos, leg_pos, neck_pos):
    # Cari skor REBA dari tabel A berdasarkan posisi tubuh bagian atas dan kaki
    row = reba_table_a[(reba_table_a['Trunk'] == trunk_pos) & 
                       (reba_table_a['Leg'] == leg_pos)]
    
    if not row.empty:
        # Pilih skor berdasarkan posisi leher
        reba_score = row[f'{neck_pos}Neck'].values[0]
        return reba_score
    else:
        return None
