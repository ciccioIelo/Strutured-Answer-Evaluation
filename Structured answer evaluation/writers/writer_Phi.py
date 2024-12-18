import json
from pathlib import Path
from promptflow.tracing import trace
import prompty
import prompty.serverless

# Definisce il percorso base della cartella contenente il file
base = Path(__file__).parent # Posizione cartella "writers"
folder = Path(__file__).parent.absolute().as_posix() # Posizione cartella generale
pathToPrompty=folder+"/writer_Phi.prompty" # Path per il prompty di Phi

@trace
def write(productContext, products, assignment):
    """
    Esegue un prompty specifico per generare una risposta basata su un contesto, prodotti e un incarico specifico.

    Args:
        productContext (str): Contesto relativo ai prodotti.
        products (dict): Informazioni sui prodotti fornite come input al modello.
        assignment (str): Richiesta specifica da risolvere.

    Returns:
        str: Risultato generato dall'LLM tramite il prompty.
    """
    # Esecuzione del prompty con i parametri forniti
    result=prompty.execute(
        pathToPrompty,
        inputs = {
            "productContext":productContext,
            "products":products, # File da usare come base per la risposta 
            "assignment":assignment # Domanda da fare all'LLM
        }
    )
    return result


# Per runnare da questo file
if __name__ == "__main__":
    from dotenv import load_dotenv

    # Carica le variabili d'ambiente dal file .env
    load_dotenv()

    # Contesto relativo ai prodotti
    productContext = "Can you use a selection of files as context?"

    # Carica il file JSON contenente i prodotti da usare come input
    products = json.loads(Path(base.parent / "json_files/find_products_output.json").read_text()) # File da cui prendere informazioni
    
    # Domanda specifica per l'LLM
    assignment = "How can I solve error 7111?" # Domanda

    # Esecuzione della funzione write
    result = write(productContext, products, assignment)

    # Stampa il risultato generato
    print(result) # Stampa risultato su console

