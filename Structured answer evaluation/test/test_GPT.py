import json
from pathlib import Path
import sys

# File di test modello gpt-4o

# Imposta il percorso base della cartella principale
base = Path(__file__).parent.parent # Path cartella base

# Utilizzo sys per importare funzioni dalle altre cartelle
sys.path.insert(0, str(base)) # Aggiunge la cartella base al path principale
sys.path.append(str(base/"index")) # Aggiunge la cartella "index" al path
sys.path.append(str(base/"writers")) # Aggiunge la cartella "writers" al path
sys.path.append(str(base/"evaluators")) # Aggiunge la cartella "evaluators" al path

# Importazione delle funzioni da altri moduli
from index import product
from writers import writer_GPT
from evaluators import eval

# Per eseguire il file di test:
# - Inserire la domanda da porre all'LLM in 'question'
# - Inserire la risposta attesa in 'ground_truth'

# Definizione della domanda e della risposta attesa
question="Qual è la sigla del PLC safety della beckhoff?"
ground_truth="La sigla PLC di sicurezza Beckhoff comunemente utilizzata nei documenti e EL6910. Questo modulo funge da PLC di sicurezza all'interno del sistema di sicurezza integrato TwinSAFE di Beckhoff."

# Imposta il contesto e l'incarico
product_context=question
assignment_context=question

# Ricerca dei file più rilevanti utilizzando l'indice
product_result=product.find_products(product_context, 20) # Trova fino a 20 file pertinenti

# Salva il risultato della ricerca dei prodotti in un file JSON
with open(Path(base / "json_files/find_products_output.json"), 'w') as outfile:
    json.dump(product_result, outfile, indent=2)

# Generazione della risposta utilizzando il modello GPT-4o
writer_result=writer_GPT.write(product_context, product_result, assignment_context) # Richiama il writer

# Calcolo le metriche di valutazione basandomi su domanda, risposta e risposta aspettata
similarity_result=eval.evaluateSimilarity(question, ground_truth, writer_result['response'])
groundedness_result=eval.evaluateGroundedness(question, outfile, writer_result['response'])
coherence_result=eval.evaluateCoherence(question, writer_result['response'])
fluency_result=eval.evaluateFluency(writer_result['response'])
relevance_result=eval.evaluateRelevance(question, writer_result['response'])

# Stampa la risposta generata e le metriche di valutazione
print(f"Risposta: {writer_result['response']}\nSources: {writer_result['sources']}")
print(f"Similarity: {similarity_result}")
print(f"Groundedness: {groundedness_result}")
print(f"Coherence: {coherence_result}")
print(f"Fluency: {fluency_result}")
print(f"Relevance: {relevance_result}")



