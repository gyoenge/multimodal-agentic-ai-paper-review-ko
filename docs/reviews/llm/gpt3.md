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

<br/> 

#### Zero-shot, one-shot, and few-shot, contrasted with traditional fine-tuning 

:::{image} ./images/gpt3-fewshot-example.png
:alt: alt text
:width: 500px
:align: center
:::

GPT-3가 집중하는 것은 "어떻게 각기 다른 Task 세팅에서 In-context learning 효과를 검증할 수 있는가"이다. 
그렇기 때문에, 기존의 전통적인 파인튜닝 방식과 In-context learning (N-shot) 방식을 포함한, 총 4가지 setting을 명확히 설정한다. 

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

#### SuperGLUE 

<!--Figure-->

GPT-3는 SuperGLUE benchmark에서 few-shot 설정만으로 fine-tuned BERT-Large와 비슷한 수준의 평균 성능을 기록했다. 다만 fine-tuned state-of-the-art 모델에는 여전히 미치지 못했으며, 특히 RTE, CB와 같은 자연어 추론(NLI) 계열 task에서 상대적으로 낮은 성능을 보였다. 이는 in-context learning이 일부 task에서는 fine-tuning에 준하는 성능을 낼 수 있지만, 모든 task에서 동등한 수준은 아님을 보여준다.

<br/>

#### Further aspects of GPT-3 

- **Emergent Arithmetic Ability**: 2~3자리 덧셈/뺄셈과 같은 산술 연산을 few-shot으로 수행하는 능력은 작은 모델에서는 거의 나타나지 않다가, 모델 크기가 일정 수준을 넘어서면 정확도가 급격히 상승한다. 175B 모델은 2자리 덧셈에서 거의 100%에 가까운 정확도를 보이는데, 사전 학습 데이터에 산술 알고리즘을 명시적으로 가르치지 않았음에도 scale만으로 이러한 능력이 발현된 대표적인 emergent capability 사례이다.
- **News Article Generation**: GPT-3가 생성한 짧은 뉴스 기사를 사람 평가자가 실제 기사와 구분하도록 했을 때, 정확도는 약 52%로 우연 수준(50%)에 가까웠다. 이는 모델이 사람이 작성한 글과 거의 구분되지 않는 수준의 자연스러운 텍스트를 생성할 수 있음을 보여준다.

<br/>

## 한계점

- **긴 텍스트 생성의 한계**: 문서 단위의 긴 텍스트를 생성할 때 주제가 반복되거나, 문장 간 모순이 발생하거나, 점차 coherence가 떨어지는 문제가 나타난다. 짧은 단위에서는 사람과 구분하기 어려운 품질을 보이지만, 길이가 길어질수록 일관성 유지가 어렵다.
- **단방향(autoregressive) 구조의 한계**: GPT-3는 왼쪽에서 오른쪽으로만 토큰을 예측하는 단방향 구조이기 때문에, 양방향 문맥을 활용해야 유리한 task(예: 빈칸 채우기, 두 문장의 의미 비교)에서는 BERT류의 bidirectional 모델 대비 상대적으로 약점을 보인다.
- **사전 학습 목적함수의 한계**: 모든 토큰을 동일한 비중으로 예측하도록 학습되기 때문에, 중요한 정보와 그렇지 않은 정보를 구분하지 못하고, 텍스트 외의 다른 modality(이미지, 영상 등)에 grounding된 지식을 학습하지 못한다.
- **샘플 효율성 문제**: 사람은 적은 텍스트만으로도 새로운 언어 task를 빠르게 학습할 수 있는 반면, GPT-3는 사전 학습 단계에서 방대한 양의 텍스트(약 300B 토큰)를 필요로 한다. in-context learning 자체는 적은 예시로 동작하지만, 그 능력을 갖추기 위한 사전 학습 비용은 매우 크다.
- **In-context learning 동작 원리의 불확실성**: few-shot 프롬프트에 주어진 예시들이 실제로 새로운 패턴을 "학습"시키는 것인지, 아니면 사전 학습 때 이미 본 적 있는 task를 "인식"하도록 유도하는 것에 가까운지 명확하지 않다.
- **모델 크기로 인한 실용적 제약**: 175B 파라미터 규모의 모델은 추론(inference) 시에도 막대한 연산 자원을 요구하여, 실제 서비스에 적용하기에는 비용과 지연시간(latency) 측면에서 부담이 크다.
- **편향(bias) 및 안전성 문제**: 인터넷 텍스트로 학습되었기 때문에 데이터에 내재된 성별, 인종, 종교 등에 대한 사회적 편향을 그대로 학습하고 재생산할 수 있으며, 이는 생성 결과의 공정성과 안전성 문제로 이어질 수 있다.

<br/>

## 인사이트

- **Scale이 곧 새로운 능력을 만든다는 실증**: 모델 구조를 거의 바꾸지 않고 파라미터 수와 데이터, 연산량만 늘렸을 뿐인데, 기존에는 명시적으로 학습시키지 않았던 in-context learning, 산술 연산 등의 능력이 특정 규모를 넘어서면서 갑자기 나타났다. 이는 이후 LLM 연구가 "더 나은 구조"보다 "더 큰 scale"을 우선적으로 탐구하게 되는 흐름의 출발점이 되었다.
- **"Pretrain + Prompt" 패러다임의 시작**: fine-tuning 없이 prompt만으로 다양한 task를 수행할 수 있음을 보여줌으로써, 이후 foundation model과 prompt engineering이라는 새로운 연구·활용 방식의 토대를 마련했다.
- **평가 방식의 전환**: zero-shot, one-shot, few-shot이라는 세 가지 평가 setting을 체계적으로 비교함으로써, task별 학습 데이터를 따로 준비하기 어려운 실제 활용 환경에 더 가까운 평가 기준을 제시했다.
- **사회적 영향에 대한 본격적인 논의 촉발**: 모델이 사람과 구분하기 어려운 텍스트를 생성할 수 있음을 보이면서, 오용 가능성, 데이터 편향, 학습에 따른 환경 비용 등 LLM의 사회적 영향에 대한 논의가 이후 연구에서 본격적으로 다뤄지는 계기가 되었다.
