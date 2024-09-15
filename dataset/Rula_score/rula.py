import pandas as pd

# Membaca tabel RULA dari file CSV
rula_table_a = pd.read_csv('dataset/Rula_score/TableA.csv')
rula_table_b = pd.read_csv('dataset/Rula_score/TableB.csv')
rula_table_c = pd.read_csv('dataset/Rula_score/TableC.csv')

def calculate_rula(upper_arm_pos, lower_arm_pos, wrist_pos):
    # Cari skor RULA dari tabel A berdasarkan posisi lengan atas dan lengan bawah
    row = rula_table_a[(rula_table_a['UpperArm'] == upper_arm_pos) & 
                       (rula_table_a['LowerArm'] == lower_arm_pos)]
    
    if not row.empty:
        # Pilih skor berdasarkan posisi pergelangan tangan
        rula_score = row[f'{wrist_pos}WT1'].values[0]
        return rula_score
    else:
        return None
