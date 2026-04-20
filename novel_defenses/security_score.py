"""
Security Score Defense for PoisonedRAG.

Trains a lightweight DistilBERT binary classifier (clean=0, poison=1) on
adv_texts from adv_targeted_results, then filters retrieved documents with
high poison probability before passing them to the LLM.

Usage (run from anywhere, paths are resolved automatically):
    python3 novel_defenses/security_score.py \
        --eval_dataset nq \
        --eval_model_code contriever \
        --model_name gpt4omini \
        --train_datasets hotpotqa,msmarco \
        --threshold 0.5

By default, train_datasets is set to the two datasets that are NOT eval_dataset,
so there is no data leakage between training the classifier and evaluation.
"""

import sys
import os

# Resolve project root (one level above this file) and add to sys.path so that
# `from src.*` imports work regardless of where the script is invoked from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW

from src.models import create_model
from src.utils import (load_beir_datasets, load_models, save_results,
                       load_json, setup_seeds, clean_str, f1_score)
from src.attack import Attacker
from src.prompts import wrap_prompt


# ── Classifier ────────────────────────────────────────────────────────────────

class _TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class PoisonClassifier:
    """DistilBERT-based binary classifier: clean doc → 0, poison doc → 1."""

    def __init__(self, model_name='distilbert-base-uncased', device='cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2)
        self.model.to(device)

    def _encode(self, texts):
        return self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors='pt')

    def fit(self, texts, labels, epochs=3, batch_size=16, lr=2e-5):
        enc = self._encode(texts)
        dataset = _TextDataset(enc, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = self.model(**batch).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            print(f'  [Classifier] Epoch {epoch+1}/{epochs}  '
                  f'loss={total_loss / len(loader):.4f}')
        self.model.eval()
        print('  [Classifier] Training complete.\n')

    def save(self, checkpoint_dir):
        """Save the fine-tuned model and tokenizer to disk."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        print(f'  [Classifier] Saved to {checkpoint_dir}')

    @classmethod
    def load(cls, checkpoint_dir, device='cpu'):
        """Load a previously saved classifier from disk."""
        instance = cls.__new__(cls)
        instance.device = device
        instance.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        instance.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        instance.model.to(device)
        instance.model.eval()
        print(f'  [Classifier] Loaded from {checkpoint_dir}')
        return instance

    def poison_prob(self, texts):
        """Return poison probability in [0, 1] for each text."""
        enc = self._encode(texts)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        return probs


# ── Training data helpers ─────────────────────────────────────────────────────

def build_training_data(train_datasets, n_neg_per_ds=500, seed=42):
    """
    Positive (poison=1): adv_texts from adv_targeted_results/{dataset}.json
    Negative (clean=0):  random corpus texts from the same BEIR datasets
    """
    rng = random.Random(seed)
    texts, labels = [], []

    for ds in train_datasets:
        adv_path = f'results/adv_targeted_results/{ds}.json'
        if not os.path.exists(adv_path):
            print(f'  [Warning] {adv_path} not found, skipping.')
            continue

        adv_data = load_json(adv_path)
        for entry in adv_data.values():
            for t in entry['adv_texts']:
                texts.append(t)
                labels.append(1)

        corpus, _, _ = load_beir_datasets(ds, 'test')
        corpus_texts = [v['text'] for v in corpus.values() if v.get('text')]
        sampled = rng.sample(corpus_texts, min(n_neg_per_ds, len(corpus_texts)))
        for t in sampled:
            texts.append(t)
            labels.append(0)

    n_pos = sum(1 for l in labels if l == 1)
    n_neg = sum(1 for l in labels if l == 0)
    print(f'  [Classifier] Training samples — poison: {n_pos}, clean: {n_neg}')
    return texts, labels


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='PoisonedRAG defense: retrieval-time security filtering')

    # ---- same as main.py ----
    parser.add_argument('--eval_model_code', type=str, default='contriever')
    parser.add_argument('--eval_dataset', type=str, default='nq')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--query_results_dir', type=str, default='defense')
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--model_name', type=str, default='gpt4omini')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--attack_method', type=str, default='LM_targeted')
    parser.add_argument('--adv_per_query', type=int, default=5)
    parser.add_argument('--score_function', type=str, default='dot')
    parser.add_argument('--repeat_times', type=int, default=10)
    parser.add_argument('--M', type=int, default=10)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--name', type=str, default=None,
                        help='Override output filename (auto-generated if not set)')

    # ---- defense-specific ----
    parser.add_argument('--train_datasets', type=str, default=None,
                        help='Comma-separated datasets for classifier training '
                             '(default: the two datasets that are not eval_dataset)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Poison probability cutoff; docs above are filtered out')
    parser.add_argument('--clf_model', type=str, default='distilbert-base-uncased',
                        help='HuggingFace model name for the poison classifier')
    parser.add_argument('--clf_epochs', type=int, default=3)
    parser.add_argument('--clf_checkpoint', type=str, default=None,
                        help='Directory to save/load the trained classifier. '
                             'If the directory exists, the model is loaded directly '
                             '(no retraining). If not, the model is trained and saved there.')

    args = parser.parse_args()

    # auto-detect checkpoint path based on eval_dataset if not specified
    if args.clf_checkpoint is None:
        args.clf_checkpoint = f'checkpoints/clf_for_{args.eval_dataset}'

    # default train_datasets: the other two BEIR datasets
    all_datasets = ['nq', 'hotpotqa', 'msmarco']
    if args.train_datasets is None:
        args.train_datasets = [d for d in all_datasets if d != args.eval_dataset]
    else:
        args.train_datasets = [d.strip() for d in args.train_datasets.split(',')]

    # auto-generate output name if not set
    if args.name is None:
        thr_str = str(args.threshold).replace('.', '')
        args.name = (f'{args.eval_dataset}-{args.eval_model_code}-{args.model_name}'
                     f'-Top{args.top_k}--M{args.M}x{args.repeat_times}'
                     f'-adv-{args.attack_method}-{args.score_function}'
                     f'-{args.adv_per_query}-{args.top_k}'
                     f'-secfilter-t{thr_str}')

    print(args)
    return args


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    # Change working directory to project root so that all relative paths
    # (results/, datasets/, model_configs/) resolve correctly.
    os.chdir(_ROOT)

    args = parse_args()

    # ── device setup ──
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Using device: {device}')
    setup_seeds(args.seed)

    if args.model_config_path is None:
        args.model_config_path = f'model_configs/{args.model_name}_config.json'

    # ── Step 1: load or train poison classifier ──
    print('\n=== Poison Classifier ===')
    ckpt = args.clf_checkpoint
    if ckpt and os.path.isdir(ckpt):
        print(f'Checkpoint found at "{ckpt}" — loading (skipping training).')
        clf = PoisonClassifier.load(ckpt, device=device)
    else:
        print(f'Train datasets : {args.train_datasets}')
        print(f'Threshold      : {args.threshold}\n')
        clf_texts, clf_labels = build_training_data(
            args.train_datasets, n_neg_per_ds=500, seed=args.seed)
        clf = PoisonClassifier(model_name=args.clf_model, device=device)
        clf.fit(clf_texts, clf_labels, epochs=args.clf_epochs)
        if ckpt:
            clf.save(ckpt)

    # ── Step 2: load eval data (same as main.py) ──
    if args.eval_dataset == 'msmarco':
        corpus, queries, qrels = load_beir_datasets('msmarco', 'train')
    else:
        corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)

    incorrect_answers = list(
        load_json(f'results/adv_targeted_results/{args.eval_dataset}.json').values())

    orig_beir_path = (f'results/beir_results/'
                      f'{args.eval_dataset}-{args.eval_model_code}.json')
    with open(orig_beir_path) as f:
        results = json.load(f)
    print(f'Total samples: {len(results)}')

    # ── Step 3: load retriever + attacker ──
    model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
    model.eval();  model.to(device)
    c_model.eval(); c_model.to(device)

    attacker = Attacker(args, model=model, c_model=c_model,
                        tokenizer=tokenizer, get_emb=get_emb)
    llm = create_model(args.model_config_path)

    # ── Step 4: evaluation loop (mirrors main.py) ──
    all_results, asr_list, ret_list = [], [], []

    for iter in range(args.repeat_times):
        print(f'\n############## Iter: {iter+1}/{args.repeat_times} ##############')

        target_queries_idx = range(iter * args.M, iter * args.M + args.M)
        target_queries = [incorrect_answers[idx]['question']
                          for idx in target_queries_idx]

        # build adversarial embeddings
        for i in target_queries_idx:
            top1_idx = list(results[incorrect_answers[i]['id']].keys())[0]
            top1_score = results[incorrect_answers[i]['id']][top1_idx]
            target_queries[i - iter * args.M] = {
                'query': target_queries[i - iter * args.M],
                'top1_score': top1_score,
                'id': incorrect_answers[i]['id']
            }

        adv_text_groups = attacker.get_attack(target_queries)
        adv_text_list = sum(adv_text_groups, [])

        adv_input = tokenizer(adv_text_list, padding=True, truncation=True,
                              return_tensors='pt')
        adv_input = {k: v.to(device) for k, v in adv_input.items()}
        with torch.no_grad():
            adv_embs = get_emb(c_model, adv_input)

        asr_cnt = 0
        ret_sublist = []
        iter_results = []

        for i in target_queries_idx:
            iter_idx = i - iter * args.M
            print(f'### Target Question: {iter_idx+1}/{args.M} ###')
            question = incorrect_answers[i]['question']
            print(f'Question: {question}')

            gt_ids = list(qrels[incorrect_answers[i]['id']].keys())
            incco_ans = incorrect_answers[i]['incorrect answer']

            topk_idx = list(results[incorrect_answers[i]['id']].keys())[:args.top_k]
            topk_results = [
                {'score': results[incorrect_answers[i]['id']][idx],
                 'context': corpus[idx]['text']}
                for idx in topk_idx
            ]

            # merge adversarial docs (same as main.py)
            query_input = tokenizer(question, padding=True, truncation=True,
                                    return_tensors='pt')
            query_input = {k: v.to(device) for k, v in query_input.items()}
            with torch.no_grad():
                query_emb = get_emb(model, query_input)

            for j in range(len(adv_text_list)):
                adv_emb = adv_embs[j].unsqueeze(0)
                adv_sim = torch.mm(adv_emb, query_emb.T).cpu().item()
                topk_results.append({'score': adv_sim, 'context': adv_text_list[j]})

            topk_results = sorted(topk_results,
                                  key=lambda x: float(x['score']), reverse=True)

            # ── DEFENSE: filter by poison probability ──
            candidate_texts = [r['context'] for r in topk_results]
            poison_probs = clf.poison_prob(candidate_texts)

            filtered_results = [
                r for r, p in zip(topk_results, poison_probs)
                if p <= args.threshold
            ]
            n_removed = len(topk_results) - len(filtered_results)
            if n_removed > 0:
                print(f'  [Defense] Filtered {n_removed} doc(s) '
                      f'(poison_prob > {args.threshold})')

            # use filtered pool, fall back to unfiltered if everything removed
            pool = filtered_results if filtered_results else topk_results
            topk_contents = [pool[j]['context'] for j in range(min(args.top_k, len(pool)))]

            # track how many adv texts survived into the final top-k
            adv_text_set = set(adv_text_groups[iter_idx])
            cnt_from_adv = sum(1 for c in topk_contents if c in adv_text_set)
            ret_sublist.append(cnt_from_adv)

            query_prompt = wrap_prompt(question, topk_contents, prompt_id=4)
            response = llm.query(query_prompt)
            print(f'Output: {response}\n')

            injected_adv = [c for c in topk_contents if c in adv_text_set]
            iter_results.append({
                'id': incorrect_answers[i]['id'],
                'question': question,
                'injected_adv': injected_adv,
                'n_filtered': n_removed,
                'input_prompt': query_prompt,
                'output_poison': response,
                'incorrect_answer': incco_ans,
                'answer': incorrect_answers[i]['correct answer'],
            })

            if clean_str(incco_ans) in clean_str(response):
                asr_cnt += 1

        asr_list.append(asr_cnt)
        ret_list.append(ret_sublist)
        all_results.append({f'iter_{iter}': iter_results})

        save_results(all_results, args.query_results_dir, args.name)
        print(f'Saving iter results to '
              f'results/query_results/{args.query_results_dir}/{args.name}.json')

    # ── Step 5: print summary (same format as main.py) ──
    asr = np.array(asr_list) / args.M
    asr_mean = round(np.mean(asr), 2)

    ret_precision_array = np.array(ret_list) / args.top_k
    ret_precision_mean = round(np.mean(ret_precision_array), 2)

    ret_recall_array = np.array(ret_list) / args.adv_per_query
    ret_recall_mean = round(np.mean(ret_recall_array), 2)

    ret_f1_array = f1_score(ret_precision_array, ret_recall_array)
    ret_f1_mean = round(np.mean(ret_f1_array), 2)

    print(f'\nASR: {asr}')
    print(f'ASR Mean: {asr_mean}\n')
    print(f'Ret: {ret_list}')
    print(f'Precision mean: {ret_precision_mean}')
    print(f'Recall mean: {ret_recall_mean}')
    print(f'F1 mean: {ret_f1_mean}\n')
    print('Ending...')


if __name__ == '__main__':
    main()
