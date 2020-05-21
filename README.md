# Introduction to T5 for Sentiment Span Extraction

This introduction [notebook](https://github.com/enzoampil/t5-intro/blob/master/t5_qa_training_pytorch_span_extraction.ipynb) is featured in Abhishek Thakur's Talks #3 webinar on [youtube](https://www.youtube.com/watch?v=4LYw_UIdd4A).

You can find the original T5 paper [here](https://arxiv.org/abs/1910.10683).

## Goals for this presentation

1. Introduce T5 and how it works
2. Explain T5's significance for the future of NLP
3. Illustrate how to use T5 for Sentiment Span Extraction


## T5 Overview

T5 is a recently released encoder-decoder model that reaches SOTA results by solving NLP problems with a text-to-text approach. This is where text is used as both an input and an output for solving all types of tasks. This was introduced in the recent paper, *Exploring the Limits of Transfer Learning with a
Unified Text-to-Text Transformer* ([paper](https://arxiv.org/pdf/1910.10683.pdf)). I've been deeply interested in this model the moment I read about it.

I believe that the combination of text-to-text as a universal interface for NLP tasks paired with multi-task learning (single model learning multiple tasks) will have a huge impact on how NLP deep learning is applied in practice.

In this presentation I aim to give a brief overview of T5, explain some of its implications for NLP in industry, and demonstrate how it can be used for sentiment span extraction on tweets. I hope this material helps you guys use T5 for your own purposes!


## Key points from T5 paper

<img src="https://drive.google.com/uc?id=15XQ-H7IdT3DVtbL7fgwYIm08OYWbB8VZ" width="700" height="300" />

1. **Treats each NLP problem as a “text-to-text” problem** - input: text, output: text

2. **Unified approach for NLP Deep Learning** - Since the task is reflected purely in the text input and output, you can use the same model, objective, training procedure, and decoding process to ANY task. Above framework can be used for any task - show Q&A, summarization, etc.

3. **Multiple NLP tasks can live in the same model** - E.g. Q&A, semantic similarity, etc. However, there is a problem called *task interference* where good results on one task can also mean worse results on another task. E.g., a good summarizer may be bad at Q&A and vice versa. All the tasks above can live in the same model, which is how it works with the released T5 models (t5-small, t5-base, etc.)

<img src="https://drive.google.com/uc?id=1-1SXVF78l3EbsuTeDUEv0hUvgfMIch3f" width="700" height="165" />

4. **New dataset: “Colossal Clean Crawled Corpus” (C4)** - a dataset consisting of ~750GB of clean English text scraped from the web. Created with a month of data from the Common Crawl corpus cleaned with a set of heuristics to filter out "unhelpful" text (e.g. offensive language, placeholder text, source code). This is a lot larger than the 13GB of data used for BERT, and 126GB of data used for XLNet.


5. **A simple denoising training objective was used for pretraining** Basically, masked language modelling but while considering contiguous masks as a single “span” to predict, and where the final prediction is an actual text sequence containing the answers (represented by “sentinel tokens”). This was compared to a language modeling pre-training objective and results consistently improved.

<img src="https://drive.google.com/uc?id=1iPG7UxZPvy6c2iwixQXYetpTzcHiKDuW" width="500" height="200" />

6. **Full encoder-decoder transformer architecture is used** - this is in contrast to previous architectures that were either encoder based (e.g. BERT), or decoder based (e.g. GPT-2). This was found effective for both generation & classification tasks.

<img src="https://drive.google.com/uc?id=1tUxL_os-pn8JTBSCY6l-9rQI0EMPC5Zz" width="400" height="500" />


## Key insight

Multiple NLP tasks can be learned by a single model since every NLP problem can be represented in a unified way - as a controllable text generation problem.

## Expected impact

Increased adoption of multi-task models like T5 due to SOTA accuracy paired with lower time, compute, & storage costs for both deployments and experiments in NLP.

## T5 for Sentiment Span Extraction (PyTorch)

1. This is a dataset from an existing Kaggle [competition](https://www.kaggle.com/c/tweet-sentiment-extraction/data) - Tweet Sentiment Extraction
2. Most of the existing model implementations use some sort of token classification task
    - The index of the beginning and ending tokens are predicted and use to *extract* the span

3. T5 is an approach that is purely *generative*, like a classic language modelling task
    - This is similar to abstractize summarization, translation, and overall text generation
    - For our data, the span is not *extracted* by predicting indices, but by generating the span from scratch

## Let's get started!

The rest of the tutorial (including the code) can be found in the T5 introduction [notebook](https://github.com/enzoampil/t5-intro/blob/master/t5_qa_training_pytorch_span_extraction.ipynb).

-----------------------------------------------------------------------------
