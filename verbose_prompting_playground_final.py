from dataclasses import dataclass
import transformers
from tqdm import tqdm
import torch
import torch.nn.functional as F
from icl.lm_apis.lm_api_base import LMForwardAPI
from icl.utils.data_wrapper import get_max_length
from icl.utils.prepare_model_and_tokenizer import load_model_and_tokenizer
from transformers import Trainer, TrainingArguments
from icl.utils.load_local import get_model_layer_num
from icl.util_classes.arg_classes import AttrArgs
from transformers import HfArgumentParser
from torch.utils.data import Dataset
from icl.analysis.attentioner_for_attribution import GPT2AttentionerManager, LlamaAttentionerManager, \
    MistralAttentionerManager


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels, _ = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "attention_mask"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, texts, tokenizer):
        data_dict = tokenize_function(texts, tokenizer)
        self.tokenizer = tokenizer
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i],
                    attention_mask=self.input_ids[i].ne(self.tokenizer.pad_token_id))


def get_start_label_ind(inputs):
    global cot
    bsz, sql = inputs['input_ids'].shape
    prefix_idxs = [29909, 29901]
    class_idx = 0
    for offset, prefix_idx in enumerate(reversed(prefix_idxs)):
        class_idx += prefix_idx * 100000 ** (offset + 1)
    input_ids = inputs['input_ids'].detach().clone() * 100000
    input_ids[:, 1:] += inputs['input_ids'][:, :-1] * 100000 * 100000
    class_pos = torch.arange(sql, device=inputs['input_ids'].device).unsqueeze(0).repeat(bsz, 1)[
        input_ids == class_idx]
    if class_pos.dim() != 1:
        class_pos = class_pos.squeeze()
    class_pos = class_pos[-1].item() + 1
    if cot:
        for p in inputs['input_ids'][:, class_pos:].squeeze():
            if p.item() == 29889:
                break
            else:
                class_pos += 1
        return class_pos + 1
    return class_pos


def tokenize_function(texts, tokenizer):
    data_dict = {"input_ids": [], "labels": []}
    for text in texts:
        new_example = tokenizer(text, padding=True,
                                max_length=get_max_length(tokenizer),
                                truncation=True,
                                return_tensors='pt')
        ind = get_start_label_ind(new_example)
        seq_len = new_example["input_ids"].shape[-1]
        temp_inputids = new_example["input_ids"]
        for i in range(ind, seq_len):
            inputids = temp_inputids[:, :i].squeeze(dim=0)
            label = temp_inputids[:, i].squeeze(dim=0)
            data_dict["input_ids"].append(inputids)
            data_dict["labels"].append(label)
    return data_dict


if __name__ == "__main__":
    hf_parser = HfArgumentParser((AttrArgs,))
    args: AttrArgs = hf_parser.parse_args_into_dataclasses()[0]
    prompt = """Q: Given a seed sentence, generate a story that includes a moral. "Once upon a time, a fox was walking through the forest."
A: Once upon a time, a fox was walking through the forest, admiring the beauty of his surroundings. He came across a pond, sparkling in the sunlight, and decided to take a drink. As he was drinking, he noticed a crocodile nearby and was stunned by its size and ferocity. When the crocodile noticed the fox, it spoke to him.\n\n"I have been here for many years and have seen many creatures come and go. I have watched them all make mistakes, as if they have never learned from their own mistakes. One day, I will tire of watching and teach them all a lesson."\n\nThe fox replied, "You must be wise to recognize this but what can a mere fox do against a powerful crocodile such as yourself?" The crocodile chuckled and said, "I am more than happy to share my wisdom with those who are willing to listen."\n\nThe fox realized that the moral of the story was to always think before acting; this is the key to avoiding making costly mistakes. He thanked the crocodile before walking away, wiser than when he had arrived.
###
Q: Compare and contrast the methods of machine learning and artificial intelligence."
A: Machine learning and artificial intelligence are closely related fields, both dealing with the automation of tasks in intelligent systems. Both techniques can help computers learn by recognizing patterns in data. However, one of the key differences between the two is that machine learning is focused on predicting outcomes while artificial intelligence focuses on replicating cognitive tasks through the use of algorithms. Machine learning involves creating an algorithm that can learn from examples to find a solution, while artificial intelligence also considers the decision-making logic behind the solution. For example, machine learning might be used to detect a certain item in an image while artificial intelligence will also factor in the context and meaning behind the image to offer a decision-making process. Artificial intelligence is able to recognize more complex patterns and variable data, while machine learning does not need to consider context, but focuses on the specific task at hand.
###
Q: Create an outline of a social media marketing strategy.
A: Outline of a Social Media Marketing Strategy \n\n1. Identify Goals and Objectives: Determine the goals for the Social Media Marketing campaign, such as increased audience engagement or increased sales.\n\n2. Research and Analyze Audience: Utilize analytical tools to research and identify target audience, including interests, preferences, and behaviors.\n\n3. Expand and Grow Audience: Utilize organic and paid methods to increase followers, likes and engagement.\n\n4. Develop and Design Content Strategy: Create a content strategy to create high-quality content that serves as a representation of the brand.\n\n5. Identify and Monitor Competitors: Research competitors and monitor their activities to identify successful marketing tactics.\n\n6. Measure and Track Performance: Analyze performance metrics and adjust strategies accordingly to maximize the reach of social media efforts.\n\n7. Network with Influencers and Industry Leaders: Identify and build relationships with influencers and industry professionals.
###
Q: {question}
A: {answer}"""

    questions = [
        "Kelian has two recipes for preparing dishes, one having 20 instructions and the second one having twice as many instructions as the first one. How many instructions does Kelian have to read to prepare the two dishes?"]
    answers = ["""1. Kelian has two recipes for preparing dishes, one having 20 instructions and the second one having twice as many instructions as the first one.

2. Kelian has to read 20 instructions to prepare the first dish.

3. Kelian has to read 40 instructions to prepare the second dish.

4. Kelian has to read 60 instructions to prepare both dishes."""]

    cot = False
    texts = []
    for q, a in zip(questions, answers):
        texts.append(prompt.format_map({"question": q, "answer": a}))

    model, tokenizer = load_model_and_tokenizer(args)

    model = LMForwardAPI(model=model, model_name=args.model_name, tokenizer=tokenizer)

    num_layer = get_model_layer_num(model=model.model, model_name=args.model_name)

    demonstrations_contexted = SupervisedDataset(texts, tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    if args.model_name in ['gpt2-xl']:
        attentionermanger = GPT2AttentionerManager(model.model)
    elif "llama" in args.model_name.lower():
        attentionermanger = LlamaAttentionerManager(model.model)
    elif "mistral" in args.model_name.lower():
        attentionermanger = MistralAttentionerManager(model.model)
    else:
        raise NotImplementedError(f"model_name: {args.model_name}")

    training_args = TrainingArguments("./output_dir", remove_unused_columns=False,
                                      # no_cuda=True,
                                      per_device_eval_batch_size=1,
                                      per_device_train_batch_size=1)
    trainer = Trainer(model=model, args=training_args, data_collator=data_collator,
                      eval_dataset=demonstrations_contexted)
    analysis_dataloader = trainer.get_eval_dataloader()

    for p in model.parameters():
        p.requires_grad = False

    total_attn_weights = []
    for idx, data in tqdm(enumerate(analysis_dataloader)):
        # data = dict_to(data, model.device)
        print(data['input_ids'].shape)
        attentionermanger.zero_grad()
        output = model(input_ids=data["input_ids"], attention_mask=data["attention_mask"])
        label = data['labels']
        print("LABEL:", tokenizer.decode(label))
        loss = F.cross_entropy(output['logits'], label)
        loss.backward()
        # sample_attn_weights = []
        for i in range(len(attentionermanger.attention_adapters)):
            saliency = attentionermanger.grad(use_abs=True)[i]
            t = saliency[:, -1, :].squeeze().sort(descending=True)
            top_values = t.values[:3]
            top_indices = t.indices[:3]
            print(saliency[:, -1, :].sum().item(), top_values, top_indices)
            for ind in top_indices:
                print(tokenizer.decode(data["input_ids"].squeeze()[ind]))
            print("#" * 80)
            # sample_attn_weights.append(attentionermanger.attention_adapters[i].attn_weights.squeeze().detach().cpu())
        # total_attn_weights.append(sample_attn_weights)
        print("*" * 80)
