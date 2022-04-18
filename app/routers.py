from fastapi import APIRouter

from .models import Sentence
from .nlp.information_extraction import NlpAlgos
from .handlers.dataset_handler import IE_brand

# router object for handling api routes
router = APIRouter()

@router.post("/postagging", response_description="POS Tagging words in a sentence")
async def pos_tag(sentence : Sentence):
    text = sentence.sentence
    pos_applied = NlpAlgos().POS_tagging(text)
    # pos_applied = json.dumps(pos_applied) 
    return pos_applied

@router.post("/dependency_graph", response_description="generates a dependency graph for a sentence")
async def generate_dependency_graph(sentence : Sentence):
    text = sentence.sentence
    nlp = NlpAlgos().nlp
    doc = nlp(text)
    dependency_graph = NlpAlgos().dependency_graph(doc)
    return dependency_graph


@router.post("/summarize", response_description="generates a text summary for a sentence")
async def generate_summary(sentence : Sentence):
    long_review  = sentence.sentence
    summarized_review = NlpAlgos().summarize(long_review)
    return summarized_review


@router.post("/sentiment", response_description="generates a text summary for a sentence")
async def generate_sentiment(brand : str):
    mean_brand_sentiment = IE_brand(brand)

    mean_brand_sentiment =  "{:.3f}".format(mean_brand_sentiment)

    return {
        "average_brand_sentiment" : mean_brand_sentiment
    }











