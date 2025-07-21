import os
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
from tqdm import tqdm
from bert_score import score as bert_score
from collections import Counter
from transformers import AutoModel, AutoTokenizer

# ==== Path configuration ====
ref_path = "test_file.jsonl"
model_names = ["base", "gemini", "qwen", "origin", "origin_cot", "gpt"]
shot_types = ["0shot", "1shot", "5shot"]

# ==== Load JSONL file ====
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return {json.loads(l)["thread_id"]: json.loads(l) for l in f}

ref_data = load_jsonl(ref_path)

# ==== Initialize BLEU & ROUGE-L ====
smooth = SmoothingFunction().method1
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# ==== Result collection list ====
result_output = []

def format_result_block(label, bleu, rouge, f1_1, f1_2, p, r, f1, d1, d2):
    return (
        f"\n📊 Summary for {label}:\n"
        f"   BLEU                            : {bleu:.4f}\n"
        f"   ROUGE-L F1                      : {rouge:.4f}\n"
        f"   best_understanding_label F1     : {f1_1:.4f}\n"
        f"   support_strategy F1             : {f1_2:.4f}\n"
        f"   BERTScore P / R / F1            : {p:.4f} / {r:.4f} / {f1:.4f}\n"
        f"   Distinct-1                      : {d1:.4f}\n"
        f"   Distinct-2                      : {d2:.4f}\n"
    )

for model in model_names:
    for shot in shot_types:
        suffix = "" if shot == "0shot" else f"_{shot}"

        model_label = f"{model}_{shot}" if shot != "0shot" else model
        print(f"\n🔍 Processing model: {model_label}")

        if model == "base":
            pred_path = f"/data/zcx/icseemo/llama3_lora_predictions{suffix}.jsonl"
        else:
            pred_path = f"/data/zcx/icseemo/llama3_lora_predictions_{model_label}.jsonl"

        try:
            pred_data = load_jsonl(pred_path)
        except FileNotFoundError:
            print(f"❌ File not found: {pred_path}, skipping")
            continue

        # Take intersection of thread_ids
        common_ids = sorted(set(ref_data.keys()) & set(pred_data.keys()))
        if not common_ids:
            print(f"⚠️ No matching data for model {model_label}, skipping")
            continue

        bleu_scores, rouge_l_scores = [], []
        labels1_true, labels1_pred = [], []
        labels2_true, labels2_pred = [], []
        bert_refs, bert_cands = [], []
        all_unigrams, all_bigrams = [], []

        for tid in tqdm(common_ids, desc=f"Computing {model_label}"):
            ref_item = ref_data[tid]
            pred_item = pred_data[tid]
            ref_text = ref_item.get("model_reply", "").strip()
            pred_text = pred_item.get("model_reply", "").strip()

            # BLEU
            ref_tokens = ref_text.split()
            pred_tokens = pred_text.split()
            bleu_scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth))

            # ROUGE-L
            rouge_scores = scorer.score(ref_text, pred_text)
            rouge_l_scores.append(rouge_scores['rougeL'].fmeasure)

            # F1 scores for classification labels
            labels1_true.append(ref_item.get("best_understanding_label", ""))
            labels1_pred.append(pred_item.get("best_understanding_label", ""))
            labels2_true.append(ref_item.get("support_strategy", ""))
            labels2_pred.append(pred_item.get("support_strategy", ""))

            # BERTScore
            bert_refs.append(ref_text)
            bert_cands.append(pred_text)

            # Distinct n-gram stats
            all_unigrams.extend(pred_tokens)
            all_bigrams.extend(list(zip(pred_tokens, pred_tokens[1:])))

        # ==== Aggregate metrics ====
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        avg_rouge = sum(rouge_l_scores) / len(rouge_l_scores)

        if len(set(labels1_true)) > 1 or len(set(labels1_pred)) > 1:
            f1_label1 = f1_score(labels1_true, labels1_pred, average="micro")
        else:
            f1_label1 = 0.0

        if len(set(labels2_true)) > 1 or len(set(labels2_pred)) > 1:
            f1_label2 = f1_score(labels2_true, labels2_pred, average="micro")
        else:
            f1_label2 = 0.0

        # ==== BERTScore with local model ====
        P, R, F1 = bert_score(
            bert_cands,
            bert_refs,
            model_type="roberta-large",
            num_layers=24,
            lang="en",
            verbose=False,
            rescale_with_baseline=False
        )

        # ==== Distinct-1 / Distinct-2 ====
        def calc_distinct(ngrams):
            total = len(ngrams)
            distinct = len(set(ngrams))
            return distinct / total if total > 0 else 0.0

        distinct_1 = calc_distinct(all_unigrams)
        distinct_2 = calc_distinct(all_bigrams)

        # ==== Print results ====
        print(f"\n📊 Summary for {model_label}:")
        print(f"   BLEU                            : {avg_bleu:.4f}")
        print(f"   ROUGE-L F1                      : {avg_rouge:.4f}")
        print(f"   best_understanding_label F1     : {f1_label1:.4f}")
        print(f"   support_strategy F1             : {f1_label2:.4f}")
        print(f"   BERTScore P / R / F1            : {P.mean():.4f} / {R.mean():.4f} / {F1.mean():.4f}")
        print(f"   Distinct-1                      : {distinct_1:.4f}")
        print(f"   Distinct-2                      : {distinct_2:.4f}")

        # ==== Save results to list ====
        block = format_result_block(
            model_label,
            avg_bleu,
            avg_rouge,
            f1_label1,
            f1_label2,
            P.mean(),
            R.mean(),
            F1.mean(),
            distinct_1,
            distinct_2
        )
        result_output.append(block)

# ==== Write results to file ====
output_path = "evaluation_results.txt"
with open(output_path, "w", encoding="utf-8") as f_out:
    f_out.write("".join(result_output))

print(f"\n✅ All evaluation results saved to: {output_path}")
