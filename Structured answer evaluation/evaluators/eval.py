from promptflow.core import Flow
from promptflow.tracing import trace
from pathlib import Path

# Imposta il percorso base della cartella corrente
base = Path(__file__).parent

# Funzione principale per calcolare le metriche di valutazione
def evaluation_metrics(request: dict, context: dict, result: str, model: str)->dict:
    """
    Calcola tutte le metriche di valutazione per il risultato generato da un modello.

    Args:
        request (dict): Dizionario contenente la 'Domanda' e la 'Ground truth'.
        context (dict): Contesto di input (es. documenti rilevanti).
        result (str): Risposta generata dal modello.
        model (str): Nome del modello utilizzato.

    Returns:
        dict: Dizionario contenente le metriche calcolate.
    """
    metrics={
        f"Risposta {model}":result,
        f"Similarity {model}": evaluateSimilarity(request['Domanda'], request['Ground truth'], result),
        f"Groundedness {model}": evaluateGroundedness(request['Domanda'], context, result),
        f"Coherence {model}": evaluateCoherence(request['Domanda'], result),
        f"Fluency {model}": evaluateFluency(result),
        f"Relevance {model}": evaluateRelevance(request['Domanda'], result)
    }
    return metrics

# Funzione per valutare la similarità tra la risposta e la "ground truth"
@trace
def evaluateSimilarity(question: str, ground_truth: str, answer: str)->str:
    """
    Valuta la similarità tra la domanda, la risposta attesa (ground truth) e la risposta generata.

    Args:
        question (str): La domanda posta.
        ground_truth (str): Risposta attesa.
        answer (str): Risposta generata.

    Returns:
        str: Risultato della valutazione di similarità.
    """
    flow = Flow.load(
        base/"similarity.prompty",
    )
    result = flow(
        question=question,
        ground_truth=ground_truth,
        answer=answer,
    )
    return result

# Funzione per valutare la "groundedness" della risposta basata sul contesto fornito
@trace
def evaluateGroundedness(question: str, context: dict, answer: str)->str:
    """
    Valuta quanto la risposta sia ancorata al contesto fornito (documenti).

    Args:
        question (str): La domanda posta.
        context (dict): Documenti utilizzati come contesto.
        answer (str): Risposta generata.

    Returns:
        str: Risultato della valutazione di groundedness.
    """
    flow = Flow.load(
        base/"groundedness.prompty",
    )
    result = flow(
        question=question,
        context=context,
        answer=answer,
    )
    return result

# Funzione per valutare la coerenza della risposta
@trace
def evaluateCoherence(question: str, answer: str)->str:
    """
    Valuta la coerenza della risposta generata.

    Args:
        question (str): La domanda posta.
        answer (str): Risposta generata.

    Returns:
        str: Risultato della valutazione di coerenza.
    """
    flow = Flow.load(
        base/"coherence.prompty",
    )
    result = flow(
        question=question,
        answer=answer,
    )
    return result

# Funzione per valutare la fluidità della risposta
@trace
def evaluateFluency(answer: str)->str:
    """
    Valuta la fluidità della risposta generata.

    Args:
        answer (str): Risposta generata.

    Returns:
        str: Risultato della valutazione di fluidità.
    """
    flow = Flow.load(
        base/"fluency.prompty",
    )
    result = flow(
        answer=answer,
    )
    return result

# Funzione per valutare la rilevanza della risposta rispetto alla domanda
@trace
def evaluateRelevance(question: str, answer: str)->str:
    """
    Valuta la rilevanza della risposta rispetto alla domanda.

    Args:
        question (str): La domanda posta.
        answer (str): Risposta generata.

    Returns:
        str: Risultato della valutazione di rilevanza.
    """
    flow = Flow.load(
        base/"relevance.prompty",
    )
    result = flow(
        question=question,
        answer=answer,
    )
    return result

    

