# GPT-3

:::{admonition} 논문 정보
:class: note

- **제목**: Language Models are Few-Shot Learners
- **저자**: Brown et al.
- **기관**: OpenAI
- **발표**: NeurIPS 2020
- **arXiv**: [링크](https://arxiv.org/abs/2005.14165)
:::

## 핵심 아이디어

GPT-3는 **모델 크기를 극단적으로 키움으로써**, 별도의 gradient update(fine-tuning) 없이 **prompt에 제시된 instruction과 소수의 example만으로 새로운 task를 수행하는 In-context Learning 능력이 자연스럽게 발현**됨을 보였다. 
이를 통해 fine-tuning 없이도 다양한 task에서 zero/one/few-shot 성능을 크게 끌어올릴 수 있음을 입증했다. 

## 방법론

GPT-3는 175 billion parameters 규모를 가지는 언어 모델이다. 
모델 아키텍처, 데이터, 학습 과정과 같은 방법론 자체는 GPT-2와 유사한 방식을 따르지만, 
GPT-3의 핵심은 거대 언어 모델이 다양한 테스크를 일반화하여 배우는 방식을 **meta learning** 관점에서 해석하고 실험으로 입증한 것이다. 

> **참고 — 아키텍처**: 모델 아키텍처는 GPT-2와 동일한 **Transformer decoder** 구조를 따른다. 175B 모델 기준 96개 레이어, 96개 attention head, $d_{\text{model}}=12288$, context window 2048 tokens로 구성된다. 또한 Sparse Transformer에서 제안된 alternating dense / locally banded sparse attention 패턴을 일부 적용한다. 
> 논문에서는 175B 모델 외에도 125M부터 175B까지 **총 8가지 모델 크기**를 함께 학습하여, scale이 성능에 미치는 영향을 체계적으로 분석한다.
> 학습 데이터는 **Common Crawl(필터링), WebText2, Books1/2, Wikipedia**로 구성되며, 총 약 300B 토큰 규모이다. 데이터 품질을 반영한 서로 다른 샘플링 가중치를 각 데이터셋에 부여한다. 

<br/>

#### Language model meta-learning 

:::{image} ./images/gpt3-metalearning.png
:alt: alt text
:width: 500px
:align: center
:::

GPT-3는 언어 모델의 학습 과정을 **meta-learning** 관점에서 해석한다.
사전 학습(outer loop)에서 모델은 방대한 텍스트 데이터로부터 광범위한 언어 패턴·지식·추론 능력을 습득한다.
추론 시(inner loop)에는 context window 내에 제공된 task description과 예시를 활용해, **gradient update 없이** 해당 task에 즉시 적응하는 In-context Learning을 수행한다. 

즉 모델은 사전 학습을 통해 "새로운 task에 빠르게 적응하는 능력 자체"를 체득하며, 이 능력은 모델 규모가 커질수록 더욱 강하게 발현된다는 것이 GPT-3의 핵심 주장이다.

#### Zero-shot, one-shot, and few-shot, contrasted with traditional fine-tuning 

:::{image} ./images/gpt3-fewshot-example.png
:alt: alt text
:width: 500px
:align: center
:::

GPT-3가 집중하는 것은 "어떻게 각기 다른 Task 세팅에서 In-context learning 효과를 검증할 수 있는가"이다. 
그렇기 때문에, 먼저 4가지 setting을 명확히 설정하는 것을 첫 번째로 제시한다. 

- **Fine-Tuning (FT)**: task별 labeled dataset으로 gradient update를 수행하는 전통적인 방식이다. 성능은 가장 높지만 task마다 별도의 데이터와 학습 비용이 필요하며, 특정 task에 과적합될 수 있다. GPT-3 논문은 FT를 직접 실험하지 않고, fine-tuning 없이 얼마나 강력한 성능을 낼 수 있는지에 집중한다.
- **Few-Shot (FS)**: task description과 K개의 (input, output) 예시를 context에 포함하여 제공하는 방식이다. K는 일반적으로 10~100개이며, context window(2048 tokens) 크기에 의해 상한이 결정된다. gradient update 없이 프롬프트만으로 새로운 task에 적응한다.
- **One-Shot (OS)**: Few-Shot에서 example이 한 개만 허용된 세팅이다. 이 세팅을 고려해야 하는 이유는 대부분의 인간과의 상호작용에서는 한 개 예시만 주어지는 경우가 많기 때문이다.
- **Zero-Shot (ZS)**: task description만 제공하고 예시 없이 바로 추론하는 방식이다. 모델이 사전 학습 지식만으로 task를 이해하고 수행해야 하므로 가장 어렵지만, 사람이 처음 보는 task를 설명만으로 수행하는 방식과 가장 유사하다.

<br/> 

## 실험 결과

#### Larger models make increasingly efficient use of in-context information 

:::{image} ./images/gpt3-result-largermodel-incontext.png
:alt: alt text
:width: 500px
:align: center
:::

- **scaling effect**: Few-shot learning 능력은 모델 크기가 커질수록 더 강한 효과를 보인다. 

<br/>

#### Aggregate performance for all 42 accuracy-denominated benchmarks 

:::{image} ./images/gpt3-result-aggregate-performance.png
:alt: alt text
:width: 500px
:align: center
:::

- **few-shot > one-shot > zero-shot**: GPT-3는 zero-shot 및 one-shot 세팅에서도 우수한 성능을 보이나, 일반적으로 few-shot > one-shot > zero-shot 순으로 성능이 향상된다. 예를 들어 CoQA 벤치마크에서 few-shot은 85.0 F1을 기록하며, zero-shot(81.5 F1)과 one-shot(84.0 F1)을 모두 상회한다. 
- **limitation of few-shot**: GPT-3의 scaling에도 불구하고 few-shot 성능이 낮게 나오는 벤치마크도 존재한다. ANLI 데이터셋의 natural language inference 태스크와 RACE/QuAC 데이터셋의 reading comprehension 태스크가 그 예시이다. 

<br/>

#### Further aspects of GPT-3 

- 

<br/>

## 한계점

(작성 예정)

<br/>

## 인사이트

(작성 예정)
