import pickle
import warnings
from dataclasses import dataclass, field
from typing import List
import os
import numpy as np
from accelerate import Accelerator
from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser
import torch
import torch.nn.functional as F

from icl.lm_apis.lm_api_base import LMForwardAPI
from icl.utils.data_wrapper import wrap_dataset, tokenize_dataset_verbose, get_max_length
from icl.utils.load_huggingface_dataset import load_huggingface_dataset_train_and_test
from icl.utils.prepare_model_and_tokenizer import load_model_and_tokenizer, \
    get_label_id_dict_for_args
from icl.utils.random_utils import set_seed
from icl.utils.other import load_args, set_gpu, sample_two_set_with_shot_per_class
from transformers import Trainer, TrainingArguments, PreTrainedModel, AutoModelForCausalLM, \
    AutoTokenizer
from icl.utils.load_local import convert_path_old, load_local_model_or_tokenizer, \
    get_model_layer_num
from icl.util_classes.arg_classes import AttrArgs
from icl.util_classes.predictor_classes import Predictor
from transformers import HfArgumentParser
from datasets import concatenate_datasets
from torch.utils.data import Dataset
from datasets.utils.logging import disable_progress_bar
import icl.analysis.attentioner_for_attribution
from icl.analysis.attentioner_for_attribution import AttentionAdapter, \
    GPT2AttentionerManager, LlamaAttentionerManager, MistralAttentionerManager
from icl.utils.other import dict_to

hf_parser = HfArgumentParser((AttrArgs,))
args: AttrArgs = hf_parser.parse_args_into_dataclasses()[0]
accelerator = Accelerator()
args.device = accelerator.device

# set_gpu(args.gpu)
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

questions = ["Kelian has two recipes for preparing dishes, one having 20 instructions and the second one having twice as many instructions as the first one. How many instructions does Kelian have to read to prepare the two dishes?"]
answers = ["""1. Kelian has two recipes for preparing dishes, one having 20 instructions and the second one having twice as many instructions as the first one.

2. Kelian has to read 20 instructions to prepare the first dish.

3. Kelian has to read 40 instructions to prepare the second dish.

4. Kelian has to read 60 instructions to prepare both dishes."""]


# prompt = """"The One" is a song by Australian singer and songwriter Kylie Minogue taken from her tenth studio album, X (2007). "The One" was written by Minogue, Richard Stannard, James Wiltshire, Russell Small, John Andersson, Johan Emmoth and Emma Holmgren, while production was handled by Stannard and Freemasons. The song was released by Parlophone in Europe and the United Kingdom, and by Warner Music in Australia and New Zealand. Originally, the song was to be accompanied with a physical release to coincide with the UK leg of the KylieX2008 tour, but was released as digital-only instead becoming Minogue's second digital single after "Over the Rainbow".\n"The One" was originally performed by dance music group Laid and Emma Holmgren, but decided to give it to Minogue. For "The One", there are two official composition mixes. The album edit is a midtempo synthpop song, while the single remix is a more upbeat dance-pop song. Freemasons decided to remix the original song for single release.
# Q: {question}
# A: {answer}"""
#
cot = False
# questions = ["Charlie had 10 stickers. He bought 21 stickers from a store in the mall and got 23 stickers for his birthday. Then Charlie gave 9 of the stickers to his sister and used 28 to decorate a greeting card. How many stickers does Charlie have left?"]
# answers = ["""Charlie had 10 stickers before he bought 21 stickers from a store in the mall. After buying the 21 stickers, Charlie had a total of 31 stickers. He then received 23 stickers for his birthday, bringing his total to 54 stickers. Charlie then gave 9 of the stickers to his sister, leaving him with 45 stickers. Finally, Charlie used 28 stickers to decorate a greeting card, leaving him with 17 stickers."""]

texts = []
for q, a in zip(questions, answers):
    texts.append(prompt.format_map({"question": q, "answer": a}))

# text = """Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).
#
# Document [1](Title: Nobel Prize in Physics) receive a diploma, a medal and a document confirming the prize amount. Nobel Prize in Physics The Nobel Prize in Physics () is a yearly award given by the Royal Swedish Academy of Sciences for those who have made the most outstanding contributions for mankind in the field of physics. It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895 and awarded since 1901; the others being the Nobel Prize in Chemistry, Nobel Prize in Literature, Nobel Peace Prize, and Nobel Prize in Physiology or Medicine. The first Nobel Prize in Physics was
# Document [2](Title: Norwegian Americans) science, Ernest Lawrence won the Nobel Prize in Physics in 1939. Lars Onsager won the 1968 Nobel Prize in Chemistry. Norman Borlaug, father of the Green Revolution, won the Nobel Peace Prize in 1970. Christian B. Anfinsen won the Nobel Prize for chemistry in 1972. Ivar Giaever won the Nobel Prize in Physics 1973. Carl Richard Hagen is noted for his work in physics. In engineering, Clayton Jacobson II is credited with the invention of the modern personal watercraft. Ole Singstad was a pioneer of underwater tunnels. Ole Evinrude invented the first outboard motor with practical commercial application, recognizable today
# Document [3](Title: Nobel Prize in Physics) Nobel Prize in Physics The Nobel Prize in Physics () is a yearly award given by the Royal Swedish Academy of Sciences for those who have made the most outstanding contributions for mankind in the field of physics. It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895 and awarded since 1901; the others being the Nobel Prize in Chemistry, Nobel Prize in Literature, Nobel Peace Prize, and Nobel Prize in Physiology or Medicine. The first Nobel Prize in Physics was awarded to physicist Wilhelm Röntgen in recognition of the extraordinary services he
# Document [4](Title: École normale supérieure (Paris)) was also awarded the Abel prize. In addition, eight "normaliens" have gone on to receive the Nobel Prize in Physics: Claude Cohen-Tannoudji, Pierre-Gilles de Gennes, Albert Fert, Alfred Kastler, Gabriel Lippmann, Louis Néel, Jean Baptiste Perrin and Serge Haroche, while other ENS physicists include such major figures as Paul Langevin, famous for developing Langevin dynamics and the Langevin equation. Alumnus Paul Sabatier won the Nobel Prize in Chemistry. A ranking of universities worldwide based on ratios of alumni to Nobel prize-winners published in 2016 by American scholars Stephen Hsu and Jonathan Wai placed ENS as the first university worldwide, far
# Document [5](Title: List of Nobel laureates in Physics) The first Nobel Prize in Physics was awarded in 1901 to Wilhelm Conrad Röntgen, of Germany, who received 150,782 SEK, which is equal to 7,731,004 SEK in December 2007.  John Bardeen is the only laureate to win the prize twice—in 1956 and 1972. Maria Skłodowska-Curie also won two Nobel Prizes, for physics in 1903 and chemistry in 1911. William Lawrence Bragg was, until October 2014, the youngest ever Nobel laureate; he won the prize in 1915 at the age of 25. Two women have won the prize: Curie and Maria Goeppert-Mayer (1963). As of 2017, the prize has been awarded
# Document [6](Title: Nobel Prize in Physics) rendered by the discovery of the remarkable rays (or x-rays). This award is administered by the Nobel Foundation and widely regarded as the most prestigious award that a scientist can receive in physics. It is presented in Stockholm at an annual ceremony on 10 December, the anniversary of Nobel's death. Through 2018, a total of 209 individuals have been awarded the prize. Only three women (1.4% of laureates) have won the Nobel Prize in Physics: Marie Curie in 1903, Maria Goeppert Mayer in 1963, and Donna Strickland in 2018. Alfred Nobel, in his last will and testament, stated that his
# Document [7](Title: E. C. George Sudarshan) had developed the breakthrough. In 2007, Sudarshan told the "Hindustan Times", "The 2005 Nobel prize for Physics was awarded for my work, but I wasn't the one to get it. Each one of the discoveries that this Nobel was given for work based on my research." Sudarshan also commented on not being selected for the 1979 Nobel, "Steven Weinberg, Sheldon Glashow and Abdus Salam built on work I had done as a 26-year-old student. If you give a prize for a building, shouldn’t the fellow who built the first floor be given the prize before those who built the second
# Document [8](Title: Svante Arrhenius) Wilhelm Ostwald, Theodore Richards) and to attempt to deny them to his enemies (Paul Ehrlich, Walther Nernst, Dmitri Mendeleev). In 1901 Arrhenius was elected to the Swedish Academy of Sciences, against strong opposition. In 1903 he became the first Swede to be awarded the Nobel Prize in chemistry. In 1905, upon the founding of the Nobel Institute for Physical Research at Stockholm, he was appointed rector of the institute, the position where he remained until retirement in 1927. He was elected a Foreign Member of the Royal Society (ForMemRS) in 1910. In 1911 he won the first Willard Gibbs Award.
#
# Q: who got the first nobel prize in physics
# A: Wilhelm Röntgen"""

# texts = [text]

model, tokenizer = load_model_and_tokenizer(args)
# model = model.half()

model = LMForwardAPI(model=model, model_name=args.model_name, tokenizer=tokenizer,
                     device=accelerator.device)


num_layer = get_model_layer_num(model=model.model, model_name=args.model_name)

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
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], attention_mask=self.input_ids[i].ne(self.tokenizer.pad_token_id))


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
            inputids = temp_inputids[:, :i]
            label = temp_inputids[:, i].squeeze(dim=0)
            data_dict["input_ids"].append(inputids)
            data_dict["labels"].append(label)
        return data_dict


demonstrations_contexted = SupervisedDataset(texts, tokenizer)

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
trainer = Trainer(model=model, args=training_args)
analysis_dataloader = trainer.get_eval_dataloader(demonstrations_contexted)


for p in model.parameters():
    p.requires_grad = False


total_attn_weights = []
for idx, data in tqdm(enumerate(analysis_dataloader)):
    data = dict_to(data, model.device)
    print(data['input_ids'].shape)
    attentionermanger.zero_grad()
    output = model(input_ids=data["input_ids"].squeeze(dim=0), attention_mask=data["attention_mask"].squeeze(dim=0))
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