import os
import torch
import logging
import warnings
from trainer import QuestionEncoder, RankingModel 

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
warnings.filterwarnings("ignore", category=FutureWarning) 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ENCODER_MODEL_PATH = "sklr/model/encoder_model.pth"
RANKING_MODEL_PATH = "sklr/model/ranking_model.pth"

num_llms = 6 
LLM_NAMES = {0: 'alpaca-13b', 1: 'claude-v1', 2: 'gpt-3.5-turbo', 3: 'gpt-4', 4: 'llama-13b', 5: 'vicuna-13b-v1.2'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

encoder_model = QuestionEncoder().to(device)
encoder_model.load_state_dict(torch.load(ENCODER_MODEL_PATH, map_location=device, weights_only=True))

embed_dim = encoder_model.encoder.config.hidden_size
ranking_model = RankingModel(embed_dim, num_llms=num_llms).to(device)
ranking_model.load_state_dict(torch.load(RANKING_MODEL_PATH, map_location=device, weights_only=True))

def predict_ranking(question):
    with torch.no_grad():
        new_embedding = encoder_model([question], device)
        llm_scores = ranking_model(new_embedding).squeeze(0) 

        sorted_indices = torch.argsort(llm_scores, descending=True)
        rankings = [0] * len(llm_scores)
        rank = 1
        for i, idx in enumerate(sorted_indices):
            if i > 0 and llm_scores[idx] == llm_scores[sorted_indices[i - 1]]:
                rankings[idx] = rankings[sorted_indices[i - 1]]
            else:
                rankings[idx] = rank
            rank += 1

        ranked_llms = [LLM_NAMES.get(idx.item(), f"unknown-{idx.item()}") for idx in sorted_indices]
        ranked_llms_with_arrow = " > ".join(ranked_llms)

        return ranked_llms, rankings, ranked_llms_with_arrow

if __name__ == "__main__":
    print("---------------------------------------------------------------------------------")
    print("Please enter the prompt!")
    question = input("Prompt: ")
    
    ranked_llms, rankings, ranked_llms_with_arrow = predict_ranking(question)

    print("LLM ranking:", ranked_llms_with_arrow)
    print("---------------------------------------------------------------------------------")