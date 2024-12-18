import json
from pathlib import Path
import sys

# File di test modello Phi

base = Path(__file__).parent.parent # Path cartella base
# Utilizzo sys per importare funzioni dalle altre cartelle
sys.path.insert(0, str(base))
sys.path.append(str(base/"index"))
sys.path.append(str(base/"writers"))
sys.path.append(str(base/"evaluators"))
from index import product
from writers import writer_Phi
from evaluators import eval

# Per eseguire il file di test mettere in 'question' la domanda da porre all'LLM e in 'ground truth' la risposta attesa

question="Qual è la sigla del PLC safety della beckhoff?"
ground_truth="La sigla PLC di sicurezza Beckhoff comunemente utilizzata nei documenti e EL6910. Questo modulo funge da PLC di sicurezza all'interno del sistema di sicurezza integrato TwinSAFE di Beckhoff."
product_context=question
assignment_context=question
product_result=product.find_products(product_context, 20) # Cerco tramite indice i file più inerenti alla domanda posta
with open(Path(base / "json_files/find_products_output.json"), 'w') as outfile:
    json.dump(product_result, outfile, indent=2)
writer_result=writer_Phi.write(product_context, product_result, assignment_context) # Risposta del modello Phi alla domanda posta
# Calcolo le metriche di valutazione basandomi su domanda, risposta e risposta aspettata
similarity_result=eval.evaluateSimilarity(question, ground_truth, writer_result)
groundedness_result=eval.evaluateGroundedness(question, outfile, writer_result)
coherence_result=eval.evaluateCoherence(question, writer_result)
fluency_result=eval.evaluateFluency(writer_result)
relevance_result=eval.evaluateRelevance(question, writer_result)
# Stampo su console la risposta e le metriche di valutazione
print(f"Risposta: {writer_result}")
print(f"Similarity: {similarity_result}")
print(f"Groundedness: {groundedness_result}")
print(f"Coherence: {coherence_result}")
print(f"Fluency: {fluency_result}")
print(f"Relevance: {relevance_result}")




