import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import torch
import json

# BERT

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics_text_class(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(
        labels,
        preds,
        average='weighted',
        zero_division=.0
    )
    precision = precision_score(
        labels,
        preds,
        average='weighted',
        zero_division=.0
    )
    recall = recall_score(
        labels,
        preds,
        average='weighted',
        zero_division=.0
    )
    accuracy = accuracy_score(
        labels,
        preds
    )
    return {
        'accuracy': round(accuracy * 100, 1),
        'precision': round(precision * 100, 1),
        'recall': round(recall * 100, 1),
        'f1': round(f1 * 100, 1)
    }


# LLMs

# LLM generation
def func_format_user(template, row):
    return template.format(
        text=row['text']
    )

def create_prompt(
        df_eval,
        func_format_user,
        messages,
        user_template
):
    # Add evaluation texts
    return [
        [
            *messages,
            {"role": "user", "content": func_format_user(user_template, row)}
        ]
        for _, row in df_eval.iterrows()
    ]


def eval_prompt(arr_input_prompt, tokenizer, model, sampling_params):
    # Model inference
    arr_tok_prompt = [
        tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=False
        )
        for prompt in arr_input_prompt
    ]
    arr_outputs = model.generate(
        prompts=arr_tok_prompt,
        sampling_params=sampling_params
    )

    arr_response = [output.outputs[0].text for output in arr_outputs]

    return arr_response


def extract_preds(df_eval, arr_preds):
    arr_t, arr_t_expl, arr_n, arr_n_expl, arr_m, arr_m_expl, arr_tnm = [], [], [], [], [], [], []
    for pred in arr_preds:
        dict_pred = json.loads(pred)
        t_label, t_expl = dict_pred['T']['label'], dict_pred['T']['explanation']
        n_label, n_expl = dict_pred['N']['label'], dict_pred['N']['explanation']
        m_label, m_expl = dict_pred['M']['label'], dict_pred['M']['explanation']
        tnm_label = t_label + n_label + m_label
        arr_t.append(t_label)
        arr_t_expl.append(t_expl)
        arr_n.append(n_label)
        arr_n_expl.append(n_expl)
        arr_m.append(m_label)
        arr_m_expl.append(m_expl)
        arr_tnm.append(tnm_label)

    return pd.DataFrame({
        "patient_id": df_eval["patient_id"].values,
        "t_label": arr_t,
        "t_explanation": arr_t_expl,
        "n_label": arr_n,
        "n_explanation": arr_n_expl,
        "m_label": arr_m,
        "m_explanation": arr_m_expl,
        "tnm_label": arr_tnm
    })
    