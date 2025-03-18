# TinyFabulist

## Introduction

### Fables: A Rich Resource for Training Small Language Models in Text Generation

Natural language is an intricate and multifaceted system, serving not only as a means of communication but also as a vehicle for imparting values, reasoning, and creativity. Fables, a timeless storytelling medium, epitomize these qualities by combining concise narratives with moral lessons and logical structures. Training language models to generate coherent, imaginative, and thematically rich fables requires them to develop a deep understanding of these essential linguistic and reasoning elements.

To generate a meaningful fable, a language model must go beyond grammar and vocabulary mastery. It must also grasp narrative flow, causal relationships, and moral reasoning. For instance, consider the prompt:

*"A rabbit challenged a tortoise to a race. Despite his speed, the rabbit learned ⟨ ⟩"*

A generative language model needs to produce a continuation that aligns with the fable's moral and thematic context, such as:
*"to never underestimate others, as persistence often triumphs over arrogance."*

This requires the model to understand not only the narrative's structure but also its embedded moral lesson.

Fables are uniquely suited for training generative capabilities in small language models (SLMs) due to their distinct characteristics. They combine narrative simplicity and depth, often comprising only a few paragraphs, which makes them manageable for SLMs with limited parameter counts. Despite their brevity, fables encapsulate profound ideas and morals, presenting a challenge for models to generate text that is both meaningful and compact. They frequently address universal human values, such as honesty, perseverance, and humility, allowing models trained on them to generalize well across different contexts and audiences.

The structured nature of fables, with a clear setup, conflict, resolution, and moral, provides an effective scaffold for SLMs to learn how to generate text with logical progression while fostering creative expression. Additionally, the vocabulary used in fables is simple yet expressive, making it ideal for focusing on essential generative skills without overwhelming models with unnecessary lexical complexity. Furthermore, generating fables requires embedding reasoning within narratives, such as establishing causal chains and demonstrating consequences, as seen in stories like "The Boy Who Cried Wolf". This demand for moral and logical reasoning enriches the model's capability to create meaningful content.

Training models on fables offers several benefits, as it equips them to produce coherent and engaging narratives with consistent themes and logical structure. The moral focus of fables allows models to generate content that is both entertaining and meaningful, while the reasoning embedded in these stories encourages the development of logical thinking and creative problem-solving. Due to their brevity and focused nature, fables are also efficient for training on modest computational resources, achieving significant improvements in generative quality.
To illustrate the potential of fable-based training, consider the following prompt for a generative task:

*"Write a story where a lion and a mouse demonstrate the value of kindness."*

A well-trained generative model might produce the following:

*"One day, a mighty lion caught a tiny mouse in his paw. 'Please let me go,' squeaked the mouse. 'Perhaps I can help you someday.' The lion laughed but released the mouse. Later, the lion was trapped in a hunter’s net. The mouse heard his roars and chewed through the ropes to free him. 'You were right,' said the lion. 'Even the smallest creatures can make a big difference.' And so, the lion and the mouse became friends, teaching that kindness is never wasted."*

This output demonstrates the model’s ability to generate a coherent and original fable with a clear moral.

In this paper, we introduce TinyFabulist, a synthetic dataset of fables meticulously crafted to capture the essential elements of moral storytelling, while reducing the complexity and breadth often associated with natural language datasets. The dataset is designed to consist exclusively of language and concepts understandable to young children, typically around 3 to 4 years old, while preserving the key characteristics of fables—structured narratives, clear morals, and anthropomorphic characters.

## Description of TinyFabulist dataset

As mentioned above, the idea behind the TinyFabulist dataset is to provide a corpus that captures the essential qualities of fables—narrative structure, moral reasoning, and thematic depth—while being smaller, less diverse, and more focused in its content. This approach mirrors how young children acquire linguistic and cognitive skills through exposure to structured stories, which combine simplicity with underlying reasoning and lessons.

To construct TinyFabulist, we utilized systematic combinatorial techniques to generate synthetic fables with clear, concise narratives. Each fable was created by combining elements drawn from predefined lists, such as characters, traits, settings, conflicts, resolutions, and morals. The dataset reflects the essence of moral storytelling while using vocabulary and constructs that are accessible to young children. A key challenge was ensuring diversity while maintaining the simplicity inherent to fables. By systematically generating all possible combinations of the predefined elements, the dataset spans a wide range of themes and scenarios, ensuring richness without overwhelming the training models.

The dataset was generated using Microsoft's phi-4 model, a language model well-suited for tasks requiring structured and creative text generation. Phi-4 was chosen for its ability to adhere to strict prompts while maintaining a high degree of creativity and narrative consistency. This is particularly important for fables, which require a balance of logical structure and imaginative storytelling. Phi-4’s strengths in generating coherent, contextually relevant text and handling explicit constraints, such as predefined characters, morals, and resolutions, made it an ideal choice for building a dataset that demands both precision and creativity.

To enhance diversity, we selected characters (e.g., Rabbit, Fox, Squirrel), traits (e.g., Brave, Cunning), and settings (e.g., Forest, River) to form the core of the fables. Each story includes a conflict (e.g., Helping someone in need, Competing for food), followed by a resolution (e.g., Reward, Punishment) and a moral (e.g., Hard work pays off, Kindness is rewarded). These combinations were used to generate thousands of unique fables. The resulting stories maintain a consistent structure, starting with a simple introduction of the characters and setting, followed by the conflict and resolution, and concluding with an explicit moral.

Using this structured template, the model is then tasked to generate new fables from varying input combinations. For example, given the input:
*Character: Rabbit*
*Trait: Brave*
*Setting: Forest*
*Conflict: Helping someone in need*
*Resolution: Reward*
*Moral: Kindness is rewarded*

The prompt generates fables such as:

*"Once there was a brave rabbit in a forest. One day, it saw a turtle stuck in a muddy pit. The rabbit used* *its strength to push a log into the pit, allowing the turtle to climb out. In gratitude, the turtle* *shared its hidden stash of berries with the rabbit.*
*Moral: Kindness is rewarded.*

Additionally, TinyFabulist supports instruction-based generation, allowing for flexibility in training and evaluation. For instance, models can be prompted with specific requirements such as a particular character or moral, enabling them to follow explicit constraints while generating coherent and relevant fables. The simplicity and structure of TinyFabulist make it an ideal dataset for training small language models, as it encourages the development of reasoning and creativity within computationally efficient frameworks. This dataset, designed for both generalization and specific task evaluation, represents a significant step in advancing research in resource-constrained generative AI.


## Evaluation

The evaluation of language models has traditionally relied on structured datasets designed to compare the model's output to predefined answers. While effective for tasks requiring deterministic responses, such approaches are less suitable for assessing creative and interpretive capabilities. To address this limitation, we propose an alternative evaluation framework tailored to small language models (SLMs) trained on generating fables. Fables serve as an excellent foundation for this purpose due to their inherent compactness, use of simple language, and ability to convey moral reasoning and narrative coherence. These characteristics align with the objectives of training and evaluating small-scale generative language models.

Our evaluation process begins by presenting the SLM with prompts based on a structured set of predefined elements, including characters, traits, settings, conflicts, resolutions, and morals. Each prompt is designed to offer a narrative scaffold for the SLM to complete, ensuring consistency while allowing for creative latitude. The SLM is tasked with generating a complete fable that adheres to the given structure, integrating the specified elements into a coherent and imaginative narrative.

To evaluate the outputs generated by the SLM, we employ GPT-4 as an automated assessor. For each generated fable, GPT-4 provides a detailed evaluation based on four key dimensions: grammar, creativity, consistency with the input prompt, and plot coherence. In addition, GPT-4 is asked to infer the likely age group of the hypothetical author of the story, offering insights into the linguistic and narrative maturity of the generated text. 

The evaluation dataset consists of fifty manually curated prompts, designed to be disjoint from the training data to avoid bias. For each prompt, the SLM generates ten completions using a temperature of 1. The resulting scores are averaged across all completions, yielding an overall performance metric for the model. This methodology ensures a robust and comprehensive assessment of the SLM's capabilities.

Our approach draws inspiration from Themis, a framework designed to evaluate moral reasoning in language models by comparing their outputs against human ethical judgments. Specifically, we adapt evaluation metrics such as grammar, creativity, and consistency from Themis, tailoring them to focus on narrative and thematic fidelity in the context of fable generation. By leveraging GPT-4's advanced capabilities, we ensure that the evaluation aligns with human-like judgments while maintaining consistency with the structured evaluation methodology outlined in Themis. For further details, see Themis: A Dataset and Benchmark for Evaluating Moral Alignment in Language Models (https://arxiv.org/pdf/2406.18365).

During training, we observe a clear correlation between evaluation scores and the model’s progress. As training loss decreases, the GPT-4 evaluation scores consistently improve, reflecting the model’s growing proficiency in generating coherent and engaging fables. Comparisons of models with varying architectures and parameter counts further reveal the impact of model design on performance, providing insights into the scalability and optimization of SLMs for this specific task.

https://arxiv.org/pdf/2303.16634 https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation

https://arxiv.org/pdf/2303.16634

https://arxiv.org/pdf/2406.18365


# Small Language Models

## SLM vs LLM

Typically, LMs that exhibit emergent abilities are classified as LLMs. However, the categorization of SLMs remains unclear. Studies vary in their contexts: some define SLMs as models with fewer than one billion parameters [199], while others consider the term “small language model” relative to the larger counterparts [163, 290, 327], with no consensus on a unified definition in the current landscape of LLMs. Research suggests SLMs for mobile devices, typically possessing around 6GB of memory, consist of sub-billion parameter models [199], whereas others classify models with up to 10 billion parameters as small, noting their lack of emergent abilities [94]. Given their use in resource-constrained environments and for specific tasks, we propose a generalized definition: Given specific tasks and resource constraints, we define SLMs as falling within a range where the lower bound is the minimum size at which the model exhibits emergent abilities for a specialized task, and the upper bound is the largest size manageable within limited resource conditions. This definition integrates various perspectives and addresses factors related to mobile computing and capability thresholds.

 https://ai.radensa.ru/wp-content/uploads/2024/11/2411.03350v1.pdf

 199-ZechunLiu,ChangshengZhao,ForrestIandola,ChenLai,YuandongTian,IgorFedorov,YunyangXiong,ErnieChang,YangyangShi,Raghuraman Krishnamoorthi, et al. 2024. Mobilellm: Optimizing sub-billion parameter language models for on-device use cases. arXiv preprint arXiv:2402.14905 (2024).

163-Jooyoung Lee, Fan Yang, Thanh Tran, Qian Hu, Emre Barut, and Kai-Wei Chang. 2024. Can Small Language Models Help Large Language Models Reason Better?: LM-Guided Chain-of-Thought. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024). 2835–2843.

290-Xuemei Tang, Jun Wang, and Qi Su. 2024. Small Language Model Is a Good Guide for Large Language Model in Chinese Entity Relation Extraction. arXiv preprint arXiv:2402.14373 (2024).

327-Yuling Wang, Changxin Tian, Binbin Hu, Yanhua Yu, Ziqi Liu, Zhiqiang Zhang, Jun Zhou, Liang Pang, and Xiao Wang. 2024. Can Small Language Models be Good Reasoners for Sequential Recommendation?. In Proceedings of the ACM on Web Conference 2024. 3876–3887.

94-Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, and Tushar Khot. 2023. Specializing smaller language models towards multi-step reasoning. In International Conference on Machine Learning. PMLR, 10421–10430.

## Best 3 Decoder-Only Transfomers

Falcon-7B, Llama-2-7B, and Mistral-7B (based on toxicity and similarity)

https://arxiv.org/pdf/2401.08491 
