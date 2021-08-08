# Papers accepted by NAACL2021
- [Papers accepted by NAACL2021](#papers-accepted-by-naacl2021)
  - [一、事件抽取](#一事件抽取)
    - [1. WEC: Deriving a Large-scale Cross-document Event Coreference dataset from Wikipedia](#1-wec-deriving-a-large-scale-cross-document-event-coreference-dataset-from-wikipedia)
    - [2. Event Representation with Sequential, Semi-Supervised Discrete Variables](#2-event-representation-with-sequential-semi-supervised-discrete-variables)
    - [3. Graph Convolutional Networks for Event Causality Identification with Rich Document-level Structures](#3-graph-convolutional-networks-for-event-causality-identification-with-rich-document-level-structures)
    - [4. Modeling Event Plausibility with Consistent Conceptual Abstraction](#4-modeling-event-plausibility-with-consistent-conceptual-abstraction)
    - [5. Document-Level Event Argument Extraction by Conditional Generation](#5-document-level-event-argument-extraction-by-conditional-generation)
    - [6. A Context-Dependent Gated Module for Incorporating Symbolic Semantics into Event Coreference Resolution](#6-a-context-dependent-gated-module-for-incorporating-symbolic-semantics-into-event-coreference-resolution)
    - [7. Event Time Extraction and Propagation via Graph Attention Networks](#7-event-time-extraction-and-propagation-via-graph-attention-networks)
    - [8. Temporal Reasoning on Implicit Events from Distant Supervision](#8-temporal-reasoning-on-implicit-events-from-distant-supervision)
    - [9. EventPlus: A Temporal Event Understanding Pipeline](#9-eventplus-a-temporal-event-understanding-pipeline)
    - [10. Counterfactual Supporting Facts Extraction for Explainable Medical Record Based Diagnosis with Graph Network](#10-counterfactual-supporting-facts-extraction-for-explainable-medical-record-based-diagnosis-with-graph-network)
  - [二、关系抽取](#二关系抽取)
    - [1. Integrating Lexical Information into Entity Neighbourhood Representations for Relation Prediction](#1-integrating-lexical-information-into-entity-neighbourhood-representations-for-relation-prediction)
    - [2.Open Hierarchical Relation Extraction](#2open-hierarchical-relation-extraction)
    - [3. A Frustratingly Easy Approach for Entity and Relation Extraction](#3-a-frustratingly-easy-approach-for-entity-and-relation-extraction)
    - [4. Distantly Supervised Relation Extraction with Sentence Reconstruction and Knowledge Base Priors](#4-distantly-supervised-relation-extraction-with-sentence-reconstruction-and-knowledge-base-priors)
    - [5. Are we there yet? Exploring clinical domain knowledge of BERT models](#5-are-we-there-yet-exploring-clinical-domain-knowledge-of-bert-models)
  - [三、命名实体识别](#三命名实体识别)
    - [1. Noisy-Labeled NER with Confidence Estimation](#1-noisy-labeled-ner-with-confidence-estimation)
    - [2. COVID-19 named entity recognition for Vietnamese](#2-covid-19-named-entity-recognition-for-vietnamese)
    - [3. Multi-Grained Knowledge Distillation for Named Entity Recognition](#3-multi-grained-knowledge-distillation-for-named-entity-recognition)
    - [4. PhoNLP: A joint multi-task learning model for Vietnamese part-of-speech tagging, named entity recognition and dependency parsing](#4-phonlp-a-joint-multi-task-learning-model-for-vietnamese-part-of-speech-tagging-named-entity-recognition-and-dependency-parsing)
    - [5. Exploring Word Segmentation and Medical Concept Recognition for Chinese Medical Texts](#5-exploring-word-segmentation-and-medical-concept-recognition-for-chinese-medical-texts)
## 一、事件抽取
### 1. WEC: Deriving a Large-scale Cross-document Event Coreference dataset from Wikipedia
Alon Eirew1,2, Arie Cattan1, Ido Dagan1

1 Bar Ilan University, Ramat-Gan, Israel 

2 Intel Labs, Israel

**Abstract**

Cross-document event coreference resolution is a foundational task for NLP applications involving multi-text processing. However, existing corpora for this task are scarce and relatively small, while annotating only modest-size clusters of documents belonging to the same topic. To complement these resources and enhance future research, we present Wikipedia Event Coreference (WEC), an efficient methodology for gathering a large-scale dataset for cross-document event coreference from Wikipedia, where coreference links are not restricted within predefined topics. We apply this methodology to the English Wikipedia and extract our large-scale WEC-Eng dataset. Notably, our dataset creation method is generic and can be applied with relatively little effort to other Wikipedia languages. To set baseline results, we develop an algorithm that adapts components of state-of-the-art models for within-document coreference resolution to the cross-document setting. Our model is suitably efficient and outperforms previously published state-of-the-art results for the task.

[[paper]](https://arxiv.org/abs/2104.05022) last revised 30 Apr 2021

### 2. Event Representation with Sequential, Semi-Supervised Discrete Variables
Mehdi Rezaee, Francis Ferraro

Department of Computer Science,University of Maryland Baltimore County,Baltimore, MD 21250 USA

**Abstract**

Within the context of event modeling and understanding, we propose a new method for neural sequence modeling that takes partially-observed sequences of discrete, external knowledge into account. We construct a sequential neural variational autoencoder, which uses Gumbel-Softmax reparametrization within a carefully defined encoder, to allow for successful backpropagation during training. The core idea is to allow semi-supervised external discrete knowledge to guide, but not restrict, the variational latent parameters during training. Our experiments indicate that our approach not only outperforms multiple baselines and the state-of-the-art in narrative script induction, but also converges more quickly.

[[paper]](https://arxiv.org/abs/2010.04361) last revised 13 Apr 2021

### 3. Graph Convolutional Networks for Event Causality Identification with Rich Document-level Structures
Minh Tran Phu: VinAI Research, Hanoi, Vietnam

Thien Huu Nguyen: Department of Computer and Information Science, University of Oregon, Eugene, Oregon, USA

**Abstract**

We study the problem of Event Causality Identification (ECI) to detect causal relation between event mention pairs in text. Although deep learning models have recently shown state-of-the-art performance for ECI, they are limited to the intra-sentence setting where event mention pairs are presented in the same sentences. This work addresses this issue by developing a novel deep learning model for document-level ECI (DECI) to accept inter-sentence event mention pairs. As such, we propose a graph-based model that constructs interaction graphs to capture relevant connections between important objects for DECI in input documents. Such interaction graphs are then consumed by graph convolutional networks to learn document context-augmented representations for causality prediction between events. Various information sources are introduced to enrich the interaction graphs for DECI, featuring discourse, syntax, and semantic information. Our extensive experiments show that the proposed model achieves state-of-the-art performance on two benchmark datasets.

[[paper]](https://aclanthology.org/2021.naacl-main.273/)

### 4. Modeling Event Plausibility with Consistent Conceptual Abstraction
Ian Porada1, Kaheer Suleman2, Adam Trischler2, Jackie Chi Kit Cheung1

1 Mila, McGill University

2 Microsoft Research Montréal

**Abstract**

Understanding natural language requires common sense, one aspect of which is the ability to discern the plausibility of events. While distributional models—most recently pre-trained, Transformer language models—have demonstrated improvements in modeling event plausibility, their performance still falls short of humans’. In this work, we show that Transformer-based plausibility models are markedly inconsistent across the conceptual classes of a lexical hierarchy, inferring that “a person breathing” is plausible while “a dentist breathing” is not, for example. We find this inconsistency persists even when models are softly injected with lexical knowledge, and we present a simple post-hoc method of forcing model consistency that improves correlation with human plausibility judgements.

[[paper]](https://aclanthology.org/2021.naacl-main.138/)

### 5. Document-Level Event Argument Extraction by Conditional Generation
Sha Li, Heng Ji, Jiawei Han

University of Illinois at Urbana-Champaign, IL, USA

**Abstract**

Event extraction has long been treated as a sentence-level task in the IE community. We argue that this setting does not match human informative seeking behavior and leads to incomplete and uninformative extraction results. We propose a document-level neural event argument extraction model by formulating the task as conditional generation following event templates. We also compile a new document-level event extraction benchmark dataset WikiEvents which includes complete event and coreference annotation. On the task of argument extraction, we achieve an absolute gain of 7.6% F1 and 5.7% F1 over the next best model on the RAMS and WikiEvents dataset respectively. On the more challenging task of informative argument extraction, which requires implicit coreference reasoning, we achieve a 9.3% F1 gain over the best baseline. To demonstrate the portability of our model, we also create the first end-to-end zero-shot event extraction framework and achieve 97% of fully supervised model’s trigger extraction performance and 82% of the argument extraction performance given only access to 10 out of the 33 types on ACE.

[[paper]](Document-Level Event Argument Extraction by Conditional Generation)

### 6. A Context-Dependent Gated Module for Incorporating Symbolic Semantics into Event Coreference Resolution
Tuan Lai, Heng Ji: University of Illinois at Urbana-Champaign

Trung Bui, Quan Hung Tran, Franck Dernoncourt, Walter Chang: Adobe Research

**Abstract**

Event coreference resolution is an important research problem with many applications. Despite the recent remarkable success of pretrained language models, we argue that it is still highly beneficial to utilize symbolic features for the task. However, as the input for coreference resolution typically comes from upstream components in the information extraction pipeline, the automatically extracted symbolic features can be noisy and contain errors. Also, depending on the specific context, some features can be more informative than others. Motivated by these observations, we propose a novel context-dependent gated module to adaptively control the information flows from the input symbolic features. Combined with a simple noisy training method, our best models achieve state-of-the-art results on two datasets: ACE 2005 and KBP 2016.

[[paper]](https://aclanthology.org/2021.naacl-main.274.pdf)

### 7. Event Time Extraction and Propagation via Graph Attention Networks
Haoyang Wen1, Yanru Qu1, Heng Ji1, Qiang Ning2∗, Jiawei Han1, Avirup Sil3, Hanghang Tong1, Dan Roth4

1 University of Illinois at Urbana-Champaign 

2 Amazon

3 IBM Research AI 4University of Pennsylvania

**Abstract**

Grounding events into a precise timeline is important for natural language understanding but has received limited attention in recent work. This problem is challenging due to the inherent ambiguity of language and the requirement for information propagation over inter-related events. This paper first formulates this problem based on a 4-tuple temporal representation used in entity slot filling, which allows us to represent fuzzy time spans more conveniently. We then propose a graph attention network-based approach to propagate temporal information over document-level event graphs constructed by shared entity arguments and temporal relations. To better evaluate our approach, we present a challenging new benchmark on the ACE2005 corpus, where more than 78% of events do not have time spans mentioned explicitly in their local contexts. The proposed approach yields an absolute gain of 7.0% in match rate over contextualized embedding approaches, and 16.3% higher match rate compared to sentence-level manual event time argument annotation.

[[paper]](https://aclanthology.org/2021.naacl-main.6/)

### 8. Temporal Reasoning on Implicit Events from Distant Supervision
Ben Zhou1,2, Kyle Richardson1, Qiang Ning3, Tushar Khot1, Ashish Sabharwal1, Dan Roth2

∗1 Allen Institute for AI 

2 University of Pennsylvania 

3 Amazon

**Abstract**

We propose TRACIE, a novel temporal reasoning dataset that evaluates the degree to which systems understand implicit events -- events that are not mentioned explicitly in natural language text but can be inferred from it. This introduces a new challenge in temporal reasoning research, where prior work has focused on explicitly mentioned events. Human readers can infer implicit events via commonsense reasoning, resulting in a more comprehensive understanding of the situation and, consequently, better reasoning about time. We find, however, that state-of-the-art models struggle when predicting temporal relationships between implicit and explicit events. To address this, we propose a neuro-symbolic temporal reasoning model, SYMTIME, which exploits distant supervision signals from large-scale text and uses temporal rules to combine start times and durations to infer end times. SYMTIME outperforms strong baseline systems on TRACIE by 5%, and by 11% in a zero prior knowledge training setting. Our approach also generalizes to other temporal reasoning tasks, as evidenced by a gain of 1%-9% on MATRES, an explicit event benchmark.

[[paper]](https://arxiv.org/abs/2010.12753) last revised 7 May 2021

### 9. EventPlus: A Temporal Event Understanding Pipeline
Mingyu Derek Ma1∗, Jiao Sun2∗, Mu Yang3, Kung-Hsiang Huang2, Nuan Wen2, Shikhar Singh2, Rujun Han2, Nanyun Peng1,2

1 Computer Science Department, University of California, Los Angeles

2 Information Sciences Institute, University of Southern California

3 Texas A&M University

**Abstract**

We present EventPlus, a temporal event understanding pipeline that integrates various state-of-the-art event understanding components including event trigger and type detection, event argument detection, event duration and temporal relation extraction. Event information, especially event temporal knowledge, is a type of common sense knowledge that helps people understand how stories evolve and provides predictive hints for future events. EventPlus as the first comprehensive temporal event understanding pipeline provides a convenient tool for users to quickly obtain annotations about events and their temporal information for any user-provided document. Furthermore, we show EventPlus can be easily adapted to other domains (e.g., biomedical domain). We make EventPlus publicly available to facilitate event-related information extraction and downstream applications.

[[paper]](https://arxiv.org/abs/2101.04922) last revised 25 Apr 2021

### 10. Counterfactual Supporting Facts Extraction for Explainable Medical Record Based Diagnosis with Graph Network
Haoran Wu1,2, Wei Chen1, Shuang Xu1, Bo Xu1,2

1 Institute of Automation, Chinese Academy of Sciences, Beijing, 100190, China

2 School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, 100049, China

**Abstract**

Providing a reliable explanation for clinical diagnosis based on the Electronic Medical Record (EMR) is fundamental to the application of Artificial Intelligence in the medical field. Current methods mostly treat the EMR as a text sequence and provide explanations based on a precise medical knowledge base, which is disease-specific and difficult to obtain for experts in reality. Therefore, we propose a counterfactual multi-granularity graph supporting facts extraction (CMGE) method to extract supporting facts from irregular EMR itself without external knowledge bases in this paper. Specifically, we first structure the sequence of EMR into a hierarchical graph network and then obtain the causal relationship between multi-granularity features and diagnosis results through counterfactual intervention on the graph. Features having the strongest causal connection with the results provide interpretive support for the diagnosis. Experimental results on real Chinese EMR of the lymphedema demonstrate that our method can diagnose four types of EMR correctly, and can provide accurate supporting facts for the results. More importantly, the results on different diseases demonstrate the robustness of our approach, which represents the potential application in the medical field.

[[paper]](https://aclanthology.org/2021.naacl-main.156/)

## 二、关系抽取
### 1. Integrating Lexical Information into Entity Neighbourhood Representations for Relation Prediction
Ian David Wood: Macquarie University and CSIRO, Australia
Stephen Wan: CSIRO, Australia

Mark Johnson: Oracle Digital Assistant

**Abstract**

Relation prediction informed from a combination of text corpora and curated knowledge bases, combining knowledge graph completion with relation extraction, is a relatively little studied task. A system that can perform this task has the ability to extend an arbitrary set of relational database tables with information extracted from a document corpus. OpenKi[1] addresses this task through extraction of named entities and predicates via OpenIE tools then learning relation embeddings from the resulting entity-relation graph for relation prediction, outperforming previous approaches. We present an extension of OpenKi that incorporates embeddings of text-based representations of the entities and the relations. We demonstrate that this results in a substantial performance increase over a system without this information.

[[paper]](Integrating Lexical Information into Entity Neighbourhood Representations for Relation Prediction)

### 2.Open Hierarchical Relation Extraction
Kai Zhang1∗, Yuan Yao1∗, Ruobing Xie2, Xu Han1, Zhiyuan Liu1†, Fen Lin2, Leyu Lin2, Maosong Sun1

1 Department of Computer Science and Technology
Institute for Artificial Intelligence, Tsinghua University, Beijing, China

Beijing National Research Center for Information Science and Technology, China

2 WeChat Search Application Department, Tencent, China

**Abstract**

Open relation extraction (OpenRE) aims to extract novel relation types from open-domain corpora, which plays an important role in completing the relation schemes of knowledge bases (KBs). Most OpenRE methods cast different relation types in isolation without considering their hierarchical dependency. We argue that OpenRE is inherently in close connection with relation hierarchies. To establish the bidirectional connections between OpenRE and relation hierarchy, we propose the task of open hierarchical relation extraction and present a novel OHRE framework for the task. We propose a dynamic hierarchical triplet objective and hierarchical curriculum training paradigm, to effectively integrate hierarchy information into relation representations for better novel relation extraction. We also present a top-down hierarchy expansion algorithm to add the extracted relations into existing hierarchies with reasonable interpretability. Comprehensive experiments show that OHRE outperforms state-of-the-art models by a large margin on both relation clustering and hierarchy expansion.

[[paper]](Open Hierarchical Relation Extraction)

### 3. A Frustratingly Easy Approach for Entity and Relation Extraction
Zexuan Zhong, Danqi Chen

Department of Computer Science, Princeton University

**Abstract**

End-to-end relation extraction aims to identify named entities and extract relations between them. Most recent work models these two subtasks jointly, either by casting them in one structured prediction framework, or performing multi-task learning through shared representations. In this work, we present a simple pipelined approach for entity and relation extraction, and establish the new state-of-the-art on standard benchmarks (ACE04, ACE05 and SciERC), obtaining a 1.7%-2.8% absolute improvement in relation F1 over previous joint models with the same pre-trained encoders. Our approach essentially builds on two independent encoders and merely uses the entity model to construct the input for the relation model. Through a series of careful examinations, we validate the importance of learning distinct contextual representations for entities and relations, fusing entity information early in the relation model, and incorporating global context. Finally, we also present an efficient approximation to our approach which requires only one pass of both entity and relation encoders at inference time, achieving an 8-16× speedup with a slight reduction in accuracy.

[[paper]](https://aclanthology.org/2021.naacl-main.5/)

### 4. Distantly Supervised Relation Extraction with Sentence Reconstruction and Knowledge Base Priors
Fenia Christopoulou1, Makoto Miwa2,3, Sophia Ananiadou1

1 National Centre for Text Mining, Department of Computer Science, The University of Manchester, United Kingdom

2 Toyota Technological Institute, Nagoya, 468-8511, Japan

3 Artificial Intelligence Research Center, National Institute of Advanced Industrial Science and Technology, Japan

**Abstract**

We propose a multi-task, probabilistic approach to facilitate distantly supervised relation extraction by bringing closer the representations of sentences that contain the same Knowledge Base pairs. To achieve this, we bias the latent space of sentences via a Variational Autoencoder (VAE) that is trained jointly with a relation classifier. The latent code guides the pair representations and influences sentence reconstruction. Experimental results on two datasets created via distant supervision indicate that multi-task learning results in performance benefits. Additional exploration of employing Knowledge Base priors into the VAE reveals that the sentence space can be shifted towards that of the Knowledge Base, offering interpretability and further improving results.

[[paper]](https://arxiv.org/abs/2104.08225) Submitted on 16 Apr 2021

### 5. Are we there yet? Exploring clinical domain knowledge of BERT models
Madhumita Sushil1, Simon Šuster2, Walter Daelemans1

1 Computational Linguistics and Psycholinguistics Research Center (CLiPS), University of Antwerp, Belgium

2 Faculty of Engineering and Information Technology, University of Melbourne

**Abstract**

We explore whether state-of-the-art BERT models encode sufficient domain knowledge to correctly perform domain-specific inference. Although BERT implementations such as BioBERT are better at domain-based reasoning than those trained on general-domain corpora, there is still a wide margin compared to human performance on these tasks. To bridge this gap, we explore whether supplementing textual domain knowledge in the medical NLI task: a) by further language model pretraining on the medical domain corpora, b) by means of lexical match algorithms such as the BM25 algorithm, c) by supplementing lexical retrieval with dependency relations, or d) by using a trained retriever module, can push this performance closer to that of humans. We do not find any significant difference between knowledge supplemented classification as opposed to the baseline BERT models, however. This is contrary to the results for evidence retrieval on other tasks such as open domain question answering (QA). By examining the retrieval output, we show that the methods fail due to unreliable knowledge retrieval for complex domain-specific reasoning. We conclude that the task of unsupervised text retrieval to bridge the gap in existing information to facilitate inference is more complex than what the state-of-the-art methods can solve, and warrants extensive research in the future.

[[paper]](https://aclanthology.org/2021.bionlp-1.5/)

## 三、命名实体识别
### 1. Noisy-Labeled NER with Confidence Estimation
Kun Liu1∗, Yao Fu2∗, Chuanqi Tan1†, Mosha Chen1, Ningyu Zhang3, Songfang Huang1, Sheng Gao4

1 Alibaba Group 

2 University of Edinburgh 

3 Zhejiang University

4 Guizhou Provincial Key Laboratory of Public Big Data, Guizhou University

**Abstract**

Recent studies in deep learning have shown significant progress in named entity recognition (NER). Most existing works assume clean data annotation, yet a fundamental challenge in real-world scenarios is the large amount of noise from a variety of sources (e.g., pseudo, weak, or distant annotations). This work studies NER under a noisy labeled setting with calibrated confidence estimation. Based on empirical observations of different training dynamics of noisy and clean labels, we propose strategies for estimating confidence scores based on local and global independence assumptions. We partially marginalize out labels of low confidence with a CRF model. We further propose a calibration method for confidence scores based on the structure of entity labels. We integrate our approach into a self-training framework for boosting performance. Experiments in general noisy settings with four languages and distantly labeled settings demonstrate the effectiveness of our method. Our code can be found at [this https URL](https://github.com/liukun95/Noisy-NER-Confidence-Estimation)

[[paper]](https://arxiv.org/abs/2104.04318) last revised 12 Apr 2021

### 2. COVID-19 named entity recognition for Vietnamese
Thinh Hung Truong, Mai Hoang Dao, Dat Quoc Nguyen

VinAI Research, Hanoi, Vietnam

**Abstract**
The current COVID-19 pandemic has lead to the creation of many corpora that facilitate NLP research and downstream applications to help fight the pandemic. However, most of these corpora are exclusively for English. As the pandemic is a global problem, it is worth creating COVID-19 related datasets for languages other than English. In this paper, we present the first manually-annotated COVID-19 domain-specific dataset for Vietnamese. Particularly, our dataset is annotated for the named entity recognition (NER) task with newly-defined entity types that can be used in other future epidemics. Our dataset also contains the largest number of entities compared to existing Vietnamese NER datasets. We empirically conduct experiments using strong baselines on our dataset, and find that: automatic Vietnamese word segmentation helps improve the NER results and the highest performances are obtained by fine-tuning pre-trained language models where the monolingual model PhoBERT for Vietnamese (Nguyen and Nguyen, 2020) produces higher results than the multilingual model XLM-R (Conneau et al., 2020). We publicly release our dataset at: [this https URL](https://github.com/VinAIResearch/PhoNER_COVID19)

[[paper]](https://arxiv.org/abs/2104.03879) Submitted on 8 Apr 2021

### 3. Multi-Grained Knowledge Distillation for Named Entity Recognition
Congying Xia1,5, Chenwei Zhang1, Tao Yang2, Yaliang Li3∗, Nan Du2, Xian Wu2, Wei Fan2, Fenglong Ma4, Philip Yu1,5

1 University of Illinois at Chicago, Chicago, IL, USA

2 Tencent Medical AI Lab, Palo Alto, CA, USA

3 Alibaba Group, Bellevue, WA, USA

4 University at Buffalo, Buffalo, NY, USA; 5Zhejiang Lab, Hangzhou, China

**Abstract**

This paper presents a novel framework, MGNER, for Multi-Grained Named Entity Recognition where multiple entities or entity mentions in a sentence could be non-overlapping or totally nested. Different from traditional approaches regarding NER as a sequential labeling task and annotate entities consecutively, MGNER detects and recognizes entities on multiple granularities: it is able to recognize named entities without explicitly assuming non-overlapping or totally nested structures. MGNER consists of a Detector that examines all possible word segments and a Classifier that categorizes entities. In addition, contextual information and a self-attention mechanism are utilized throughout the framework to improve the NER performance. Experimental results show that MGNER outperforms current state-of-the-art baselines up to 4.4% in terms of the F1 score among nested/non-overlapping NER tasks.

[[paper]](https://arxiv.org/abs/1906.08449)

### 4. PhoNLP: A joint multi-task learning model for Vietnamese part-of-speech tagging, named entity recognition and dependency parsing
Linh The Nguyen, Dat Quoc Nguyen

VinAI Research, Hanoi, Vietnam

**Abstract**

We present the first multi-task learning model -- named PhoNLP -- for joint Vietnamese part-of-speech (POS) tagging, named entity recognition (NER) and dependency parsing. Experiments on Vietnamese benchmark datasets show that PhoNLP produces state-of-the-art results, outperforming a single-task learning approach that fine-tunes the pre-trained Vietnamese language model PhoBERT (Nguyen and Nguyen, 2020) for each task independently. We publicly release PhoNLP as an open-source toolkit under the Apache License 2.0. Although we specify PhoNLP for Vietnamese, our PhoNLP training and evaluation command scripts in fact can directly work for other languages that have a pre-trained BERT-based language model and gold annotated corpora available for the three tasks of POS tagging, NER and dependency parsing. We hope that PhoNLP can serve as a strong baseline and useful toolkit for future NLP research and applications to not only Vietnamese but also the other languages. Our PhoNLP is available at: [this https URL](https://github.com/VinAIResearch/PhoNLP)

[[paper]](https://arxiv.org/abs/2101.01476)  last revised 8 Apr 2021

### 5. Exploring Word Segmentation and Medical Concept Recognition for Chinese Medical Texts
Yang Liu♠♦, Yuanhe Tian♥, Tsung-Hui Chang♠♦, Song Wu♣4, Xiang Wan♦, Yan Song♠♦†

♠ The Chinese University of Hong Kong (Shenzhen)

♥ University of Washington

♣ PingHu Hospital of Shenzhen University

4 Shenzhen Hospital of Shanghai University of Traditional Chinese Medicine

♦ Shenzhen Research Institute of Big Data

**Abstract**

Chinese word segmentation (CWS) and medical concept recognition are two fundamental tasks to process Chinese electronic medical records (EMRs) and play important roles in downstream tasks for understanding Chinese EMRs. One challenge to these tasks is the lack of medical domain datasets with high-quality annotations, especially medical-related tags that reveal the characteristics of Chinese EMRs. In this paper, we collected a Chinese EMR corpus, namely, ACEMR, with human annotations for Chinese word segmentation and EMR-related tags. On the ACEMR corpus, we run well-known models (i.e., BiLSTM, BERT, and ZEN) and existing state-of-the-art systems (e.g., WMSeg and TwASP) for CWS and medical concept recognition. Experimental results demonstrate the necessity of building a dedicated medical dataset and show that models that leverage extra resources achieve the best performance for both tasks, which provides certain guidance for future studies on model selection in the medical domain.

[[paper]](https://aclanthology.org/2021.bionlp-1.23/)
