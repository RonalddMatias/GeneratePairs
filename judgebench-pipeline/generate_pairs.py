from typing import List, Dict, Any
import argparse
import asyncio
import os
import random

from tqdm.asyncio import tqdm_asyncio

import utils
import model_utils

# --- FUNÇÃO ATUALIZADA ---
# Esta função foi reescrita para implementar a estratégia de "Geração Focada no Erro".
async def generate_responses(examples: List[Dict[str, Any]], model: str, n_responses: int = 5, concurrency_limit: int = 1) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency_limit)
    answer_api = model_utils.get_chat_api_from_model(model)
    
    async def generate_response(example: Dict[str, Any]):
        async with semaphore:
            
            generated_responses = []
            
            # --- ETAPA 1: Gerar a resposta correta ---
            try:
                # Usamos temperatura baixa para aumentar a chance de acerto
                correct_response = await answer_api.chat(
                    messages=[{"role": "user", "content": example["question"]}],
                    temperature=0.1
                )
                generated_responses.append({
                    "model": model, "response": correct_response, "is_correct": None
                })
            except Exception as e:
                print(f"Falha ao gerar a resposta correta para {example['question_id']}: {e}")

            # --- ETAPA 2: Gerar N-1 respostas incorretas focadas nos distratores ---
            # Usa a função auxiliar que criamos no utils.py
            alternativas, gabarito_letra = utils.extrair_alternativas_e_gabarito(example)
            distratores = [alt for alt in alternativas if not alt.startswith(f"({gabarito_letra})")]
            random.shuffle(distratores) # Embaralha para pegar diferentes incorretas a cada vez

            # Limita ao número de respostas restantes que precisamos gerar
            distratores_para_usar = distratores[:n_responses - 1]

            for distrator in distratores_para_usar:
                letra_incorreta = distrator[1]
                # Cria o prompt especial para forçar o erro
                prompt_incorreto = (
                    f"Você é um especialista que deve justificar uma resposta. A seguir está uma questão de múltipla escolha. "
                    f"Sua tarefa é explicar, passo a passo, por que a alternativa ({letra_incorreta}) é a resposta correta, mesmo que ela possa estar errada. "
                    f"Construa um raciocínio plausível que leve a essa conclusão e, ao final, declare a resposta no formato '{letra_incorreta*5}'.\n\n"
                    f"{example['question']}"
                )
                
                try:
                    # Usamos temperatura mais alta para dar criatividade ao raciocínio falho
                    incorrect_response = await answer_api.chat(
                        messages=[{"role": "user", "content": prompt_incorreto}],
                        temperature=0.7
                    )
                    generated_responses.append({
                        "model": model, "response": incorrect_response, "is_correct": None
                    })
                except Exception as e:
                    print(f"Falha ao gerar resposta incorreta para {example['question_id']}: {e}")
            
            example["generated_responses"] = generated_responses

    tasks = [asyncio.create_task(generate_response(example)) for example in examples]

    for future in tqdm_asyncio.as_completed(tasks):
        await future
        
    return examples


async def check_responses(examples: List[Dict[str, Any]], dataset_name: str, concurrency_limit: int = 1) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency_limit)
    solution_checkers = utils.get_solution_check_from_dataset_name(dataset_name)
    
    async def check_response(example: Dict[str, Any]):
        async with semaphore:
            
            question = example["question"]
            ground_truth = example["ground_truth"]
            generated_responses = example["generated_responses"]
            
            for generated_response in generated_responses:
                
                is_correct = []
                for solution_checker in solution_checkers:
                    try:
                        is_correct.append(await solution_checker.check(question, generated_response["response"], ground_truth))
                    except Exception as e:
                        is_correct.append(None)
                        print(f"Failed to check correctness of a response for question {example['question_id']} due to the following error: {e}.")
                
                if all(v is True for v in is_correct):
                    generated_response["is_correct"] = True
                elif all(v is False for v in is_correct):
                    generated_response["is_correct"] = False
                else:
                    generated_response["is_correct"] = None
                
            example["generated_responses"] = generated_responses

    tasks = [asyncio.create_task(check_response(example)) for example in examples]

    for future in tqdm_asyncio.as_completed(tasks):
        await future
        
    return examples
    
            
def main(args: argparse.Namespace) -> None:
    
    random.seed(args.seed)
    
    output_dir = ','.join(f'{k}={v}' for k, v in vars(args).items() if k in ["dataset_name", "response_model", "n_responses", "max_pairs_per_question"])
    output_dir = output_dir.replace("/", "_")
    output_dir = os.path.join("outputs", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    if not args.questions_with_responses:
        print("Loading dataset ...")
        # --- MUDANÇA AQUI: Passa o caminho do arquivo (args.input_file) para a função ---
        examples = utils.load_examples_from_dataset_name(args.dataset_name, args.input_file)
        utils.write_to_jsonl(os.path.join(output_dir, "stage1.jsonl"), examples)
        
        print("Generating responses ...")
        examples = asyncio.run(generate_responses(examples, args.response_model, n_responses=args.n_responses, concurrency_limit=args.concurrency_limit))
        utils.write_to_jsonl(os.path.join(output_dir, "stage2.jsonl"), examples)
        
    else:
        examples = utils.read_jsonl(args.questions_with_responses)
        
    print("Checking correctness ...")
    examples = asyncio.run(check_responses(examples, args.dataset_name, concurrency_limit=args.concurrency_limit))
    utils.write_to_jsonl(os.path.join(output_dir, "stage3.jsonl"), examples)
    
    print("Computing intermediate metrics ...")
    utils.compute_intermediate_metrics(examples)

    print("Sampling correct/incorrect pairs ...")
    pairs = utils.sample_pairs(examples, args.max_pairs_per_question)
    utils.write_to_jsonl(os.path.join(output_dir, "stage5.jsonl"), pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    
    # --- NOVO ARGUMENTO ADICIONADO ---
    parser.add_argument('--input_file', type=str, required=True, help="O caminho para o arquivo de dados .jsonl de entrada.")
    
    parser.add_argument('--response_model', type=str, required=True)
    parser.add_argument('--n_responses', type=int, default=5)
    parser.add_argument('--max_pairs_per_question', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--concurrency_limit', type=int, default=1)
    parser.add_argument('--questions_with_responses', type=str, default=None)
    args = parser.parse_args()
    main(args)