import os
import json
from typing import Dict, List
from promptflow.tracing import trace
from promptflow.core import Flow
from openai import AzureOpenAI
from dotenv import load_dotenv
from pathlib import Path
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    VectorizedQuery,
    QueryType,
    QueryCaptionType,
    QueryAnswerType,
)
from azure.core.credentials import AzureKeyCredential

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Definizione del percorso base
base = Path(__file__).parent

# Caricamento delle variabili di configurazione per Azure OpenAI e Azure Search
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_VERSION = "2023-12-01-preview"
AZURE_OPENAI_DEPLOYMENT = "text-embedding-ada-002"
AZURE_SEARCH_SERVICE= os.getenv("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX= os.getenv("AZURE_SEARCH_INDEX")

# Funzione per generare embeddings utilizzando Azure OpenAI
@trace
def generate_embeddings(queries: List[str]) -> str:
    """
    Genera embeddings per una lista di query usando il servizio Azure OpenAI.
    
    Args:
        queries (List[str]): Una lista di stringhe (query).
    
    Returns:
        List[Dict[str, any]]: Una lista di dizionari contenenti la query originale e il relativo embedding.
    """
    client = AzureOpenAI(
        api_version=AZURE_OPENAI_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        api_key=AZURE_OPENAI_KEY,
    )

    # Genera embeddings tramite Azure OpenAI
    embeddings = client.embeddings.create(input=queries, model=AZURE_OPENAI_DEPLOYMENT)
    embs = [emb.embedding for emb in embeddings.data]

    # Crea una lista di dizionari con query e embedding associati
    items = [{"item": queries[i], "embedding": embs[i]} for i in range(len(queries))]

    return items


# Funzione per recuperare prodotti dall'indice di Azure Search utilizzando embeddings
@trace
def retrieve_products(items: List[Dict[str, any]], index_name: str, top: int) -> str:
    """
    Recupera documenti rilevanti dall'indice di Azure Search basandosi sugli embeddings forniti.
    
    Args:
        items (List[Dict[str, any]]): Lista di query e embeddings.
        index_name (str): Nome dell'indice Azure Search.
        top (int): Numero massimo di risultati da restituire.
    
    Returns:
        List[Dict[str, any]]: Una lista di documenti rilevanti.
    """
    # Creazione del client Azure Search
    search_client = SearchClient(
        endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
        index_name=index_name,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
    )

    products = [] # Lista per accumulare i prodotti trovati
    for item in items:
        vector_query = VectorizedQuery(
            vector=item["embedding"], fields="embedding"
        )

        # Esecuzione della ricerca sull'indice Azure Search
        results = search_client.search(
            search_text=item["item"],
            vector_queries=[vector_query],
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name="default",
            query_caption=QueryCaptionType.EXTRACTIVE,
            query_answer=QueryAnswerType.EXTRACTIVE,
            top=top,
        )

        # Estrazione dei documenti rilevanti
        docs = [
            {
                "id": doc["id"],
                "content": doc["content"],
                "sourcepage": doc["sourcepage"],
                "sourcefile": doc["sourcefile"],
                "storageUrl": doc["storageUrl"],
            }
            for doc in results
        ]

        # Rimuove duplicati basati sull'ID
        products.extend([i for i in docs if i["id"] not in [x["id"] for x in products]])

    return products


# Funzione principale per trovare i prodotti
@trace
def find_products(context: str, top: int) -> Dict[str, any]:
    """
    Trova documenti rilevanti basati su una query fornita.
    
    Args:
        context (str): Contesto della ricerca (es. domanda dell'utente).
        top (int): Numero massimo di documenti da restituire.
    
    Returns:
        Dict[str, any]: Documenti rilevanti trovati nell'indice Azure Search.
    """

    """
    #flow = Flow.load(base / "product.prompty")
    #queries = flow(context=context)
    #qs = json.loads(queries)
    #items = generate_embeddings(qs)
    """
    # Scommentare la parte precedente e commentare fino a "Retrieve products" se per la ricerca sull'indice si vuole 
    # scomporre la domanda in 5 query significative
    
    # Utilizza direttamente la query originale
    queries=[context] 

    # Genera embeddings per la query
    items = generate_embeddings(queries)   
    
    # Retrieve products
    products = retrieve_products(items, AZURE_SEARCH_INDEX, top)

    return products


# Per runnare da questo file
if __name__ == "__main__":
    # Query di esempio
    context = "What's error 4115?" # Domanda di esempio

    # Recupera i documenti rilevanti
    answer = find_products(context , 20) # 20 documenti al massimo
    
    # Salvataggio dei risultati in un file JSON
    with open(base.parent/'json_files/find_products_output.json', 'w') as outfile:
        json.dump(answer, outfile, indent=2)

    # Stampa i risultati
    print(json.dumps(answer, indent=2))