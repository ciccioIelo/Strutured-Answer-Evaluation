import json
from pathlib import Path
from index import product
from writers import writer_GPT
from writers import writer_Phi
from evaluators import eval
import pandas as pd

def merge_metrics(*results_dicts):
    models = ["gpt-4o", "Phi"]  # Lista dei modelli
    metrics = ["Risposta", "Similarity", "Groundedness", "Coherence", "Fluency", "Relevance"]  # Lista delle metriche

    """
    Unisce dizionari di metriche separando quelle per modello (gpt-4o e Phi).
    :param results_dicts: Dizionari multipli di metriche.
    :return: Dizionario organizzato per metrica e modello.
    """
    merged = {}  # Dizionario finale

    # Unisci i dizionari di input in un solo dizionario
    combined = {}
    for result in results_dicts:
        combined.update(result)
    
    # Scorri tutte le metriche e i modelli
    for metric in metrics:
        for model in models:
            # Crea la chiave per il dizionario finale usando la metrica e il modello
            key = f"{metric} {model}"
            
            # Cerca il valore corrispondente nel dizionario combinato
            # Se la chiave esiste, lo aggiungi nel dizionario finale
            if key in combined:
                merged[key] = combined[key]
            else:
                merged[key] = None  # Se non esiste la metrica per quel modello, metti un valore None

    return merged



base = Path(__file__).parent
excel_file="xlsx/Marchesini_Q&GT.xlsx" # Nome del file excel di input da cui prende domanda e ground truth
json_file="json_files/request.json" # File JSON in cui viene convertito il file excel di input
df=pd.read_excel(base/excel_file) # Leggi il file Excel in un DataFrame pandas
df.to_json(base/json_file, orient="records",indent=4) # Converti il DataFrame in JSON
with open(Path(base / json_file), 'r') as file:
    data = json.load(file) # Carica il JSON in memoria

data1=[] # Lista per memorizzare i risultati delle domande

# Per ogni domanda nel file JSON
for request in data:
    product_context=request['Domanda']
    assignment_context=request['Domanda']
    try:
        # Ricerca di 20 file inerenti tramite indice (prodotti correlati alla domanda)
        product_result=product.find_products(product_context , 20)

        with open(Path(base / "json_files/find_products_output.json"), 'w') as outfile:
            json.dump(product_result, outfile, indent=2)

        # Generazione delle risposte dai modelli GPT-4o e Phi  
        gpt_result=writer_GPT.write(product_context, product_result, assignment_context) # Risposta usando gpt-4o
        phi_result=writer_Phi.write(product_context, product_result, assignment_context) # Risposta usando Phi-3-small

        # Calcolo delle metriche di valutazione per ciascun modello
        metrics_gpt=eval.evaluation_metrics(request, outfile, gpt_result['response'], "gpt-4o") # Calcolo metriche di valutazione della risposta di gpt-4o
        metrics_phi=eval.evaluation_metrics(request, outfile, phi_result, "Phi") # Calcolo metriche di valutazione della risposta di gpt-4o

        # Creazione dell'elemento JSON relativo alla domanda corrente
        # Parametri uguali per ogni modello (passati in input)
        new_output={
            "Numero":request['Nr'],
            "Domanda":request['Domanda'],
            "Ground Truth":request['Ground truth'],
        }

        # Aggiungo le metriche ordinandole
        all_metrics = merge_metrics(metrics_gpt,metrics_phi)
        new_output.update(all_metrics)

        # Aggiungo l'elemento JSON relativo alla domanda al file di output JSON
        if isinstance(data1, list):
            data1.append(new_output)
        
        # Stampa dei risultati su console
        print(gpt_result)
        print(phi_result)

    except Exception as e:
        print(str(e))

# Scrivi i risultati finali in un file JSON
with open (Path(base / "json_files/output.json"), 'w') as file:
    json.dump(data1, file, indent=4)
    
df1=pd.read_json(Path(base / "json_files/output.json")) # Leggi il JSON di output in un DataFrame


# Formattazione file excel di output per una migliore leggibilità

# Raggruppa ogni n colonne, tralasciando le prime 3 (numero, domanda, ground truth)
group_size = 2 #numero modelli
num_cols = len(df1.columns)

# Scrittura dei risultati finali su un file Excel
with pd.ExcelWriter(base/"xlsx/output.xlsx", engine="xlsxwriter") as writer:
    df1.to_excel(writer, index=False, sheet_name="Foglio1")
    workbook = writer.book
    worksheet = writer.sheets["Foglio1"]

    # Imposta larghezza colonne e testo a capo per migliorare la leggibilità
    wrap_format = workbook.add_format({"text_wrap": True, "align": "center", "valign": "vcenter"})
    for idx, col in enumerate(df1.columns):
        max_width = max(df1[col].astype(str).map(len).max(), len(col)) + 2
        adjusted_width = min(max_width, 50) # Limita la larghezza della colonna
        worksheet.set_column(idx, idx, adjusted_width, wrap_format)

    # Formato per i bordi
    left_border_format = workbook.add_format({"left": 2})    # Bordo spesso a sinistra
    right_border_format = workbook.add_format({"right": 2})  # Bordo spesso a destra

    # Applica i bordi solo alle colonne di inizio e fine dei gruppi
    for start_idx in range(3, num_cols, group_size):
        # Colonna sinistra del gruppo
        worksheet.conditional_format(1, start_idx, len(df1), start_idx, {
            "type": "no_blanks",
            "format": left_border_format
        })
        # Colonna destra del gruppo
        end_idx = min(start_idx + group_size - 1, num_cols - 1)
        worksheet.conditional_format(1, end_idx, len(df1), end_idx, {
            "type": "no_blanks",
            "format": right_border_format
        })

        # Formattazione per la media (grassetto e centrato)
    mean_format = workbook.add_format({"bold": True, "align": "center", "valign": "vcenter", "num_format": "0.00"})

    # Scrivi la media sotto ogni colonna numerica
    last_row = len(df1) + 1  # La riga successiva all'ultima riga di dati
    for col_idx, col_name in enumerate(df1.columns):
        if pd.api.types.is_numeric_dtype(df1[col_name]):  # Controlla se la colonna è numerica
            mean_value = df1[col_name].mean() # Calcola la media della colonna
            worksheet.write_number(last_row, col_idx, mean_value, mean_format) # Scrivi la media nel foglio




