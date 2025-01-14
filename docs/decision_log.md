# Small Language Models

## SLM vs LLM

Typically, LMs that exhibit emergent abilities are classified as LLMs. However, the categorization of SLMs remains unclear. Studies vary in their contexts: some define SLMs as models with fewer than one billion parameters [199], while others consider the term “small language model” relative to the larger counterparts [163, 290, 327], with no consensus on a unified definition in the current landscape of LLMs. Research suggests SLMs for mobile devices, typically possessing around 6GB of memory, consist of sub-billion parameter models [199], whereas others classify models with up to 10 billion parameters as small, noting their lack of emergent abilities [94]. Given their use in resource-constrained environments and for specific tasks, we propose a generalized definition: Given specific tasks and resource constraints, we define SLMs as falling within a range where the lower bound is the minimum size at which the model exhibits emergent abilities for a specialized task, and the upper bound is the largest size manageable within limited resource conditions. This definition integrates various perspectives and addresses factors related to mobile computing and capability thresholds.

 https://ai.radensa.ru/wp-content/uploads/2024/11/2411.03350v1.pdf

 199-ZechunLiu,ChangshengZhao,ForrestIandola,ChenLai,YuandongTian,IgorFedorov,YunyangXiong,ErnieChang,YangyangShi,Raghuraman Krishnamoorthi, et al. 2024. Mobilellm: Optimizing sub-billion parameter language models for on-device use cases. arXiv preprint arXiv:2402.14905 (2024).

163-Jooyoung Lee, Fan Yang, Thanh Tran, Qian Hu, Emre Barut, and Kai-Wei Chang. 2024. Can Small Language Models Help Large Language Models Reason Better?: LM-Guided Chain-of-Thought. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024). 2835–2843.

290-Xuemei Tang, Jun Wang, and Qi Su. 2024. Small Language Model Is a Good Guide for Large Language Model in Chinese Entity Relation Extraction. arXiv preprint arXiv:2402.14373 (2024).

327-Yuling Wang, Changxin Tian, Binbin Hu, Yanhua Yu, Ziqi Liu, Zhiqiang Zhang, Jun Zhou, Liang Pang, and Xiao Wang. 2024. Can Small Language Models be Good Reasoners for Sequential Recommendation?. In Proceedings of the ACM on Web Conference 2024. 3876–3887.

94-Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, and Tushar Khot. 2023. Specializing smaller language models towards multi-step reasoning. In International Conference on Machine Learning. PMLR, 10421–10430.

##Best 3 Decoder-Only Transfomers

Falcon-7B, Llama-2-7B, and Mistral-7B (based on toxicity and similarity)

https://arxiv.org/pdf/2401.08491 

## Evaluation

https://arxiv.org/pdf/2303.16634 https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation

https://arxiv.org/pdf/2303.16634

https://arxiv.org/pdf/2406.18365