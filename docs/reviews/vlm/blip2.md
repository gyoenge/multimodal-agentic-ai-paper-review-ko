# BLIP-2

:::{admonition} 논문 정보
:class: note

- **제목**: BLIP-2: Bootstrapping Language-Image Pre-Training with Frozen Image Encoders and Large Language Models
- **저자**: Li et al.
- **기관**: Salesforce
- **발표**: ICML 2023
- **arXiv**: [링크](https://arxiv.org/abs/2301.12597)
:::

## 핵심 아이디어

BLIP-2는 **이미 잘 학습된 모델을 다시 학습시키지 않는다**는 발상에서 출발한다. 강력한 이미지 인코더와 거대 언어 모델을 **모두 frozen 상태로 두고**, 그 사이를 잇는 가벼운 모듈 하나만 학습시킨다.

[BLIP](blip.md)을 포함한 기존 vision-language pre-training(VLP)은 대부분 **end-to-end 학습**을 전제한다. 그런데 모델 규모가 커질수록 이 방식의 비용은 감당하기 어려워진다. 이미지 인코더와 언어 모델을 매번 처음부터 함께 학습시키는 것은, 각 모달리티에서 이미 최고 수준에 도달한 단일 모달 모델들이 존재하는 상황에서 **명백한 낭비**이다.

그렇다면 기존 모델을 그대로 가져다 쓰면 되지 않을까. 문제는 여기서 발생하는 **modality gap**이다. 언어 모델은 사전 학습 과정에서 **이미지를 단 한 번도 본 적이 없다.** 게다가 두 모델을 frozen으로 두면 정렬을 학습할 여지 자체가 사라지므로, 단순히 이미지 feature를 LLM에 밀어 넣는 것만으로는 작동하지 않는다.

BLIP-2는 이 간극을 **Q-Former(Querying Transformer)** 라는 학습 가능한 경량 모듈로 메운다. 그리고 이를 **두 단계에 걸쳐 사전 학습**한다. 먼저 frozen 이미지 인코더로부터 **언어와 정렬된 시각 표현을 학습**하고(1단계), 그다음 그 표현을 frozen LLM에 연결해 **생성 능력을 학습**한다(2단계).

그 결과 BLIP-2는 학습 가능한 파라미터가 **약 1.9억 개**에 불과하면서도, zero-shot VQAv2에서 **Flamingo80B를 8.7% 앞선다.** Flamingo의 학습 파라미터가 102억 개인 것과 비교하면 **54배 적은 규모**이다.

## 방법론

### Q-Former

<!--Figure: Figure 2 (Model architecture of Q-Former)-->

Q-Former는 frozen 이미지 인코더와 frozen LLM 사이에 놓이는 **약 188M 규모의 경량 transformer**로, BLIP-2에서 유일하게 학습되는 부분이다.

핵심은 입력이 이미지가 아니라 **학습 가능한 query embedding 집합**이라는 점이다. **32개의 query**(각 768차원)가 모델 파라미터의 일부로서 함께 학습되며, 이들이 이미지로부터 무엇을 뽑아낼지를 스스로 익힌다.

Q-Former는 **self-attention 레이어를 공유하는 두 개의 서브모듈**로 구성된다.

- **Image Transformer**: query들이 frozen 이미지 feature와 상호작용하여 시각 정보를 추출한다. **cross-attention 레이어는 transformer 블록마다가 아니라 한 블록 걸러 하나씩 삽입**된다.
- **Text Transformer**: 텍스트 인코더와 디코더 역할을 모두 수행한다.

query들은 self-attention을 통해 서로 정보를 주고받고, cross-attention을 통해 이미지를 참조하며, 공유된 self-attention 레이어를 통해 텍스트와도 상호작용할 수 있다. 초기화는 사전 학습된 **BERT-base** 가중치로 하되, cross-attention 레이어만 랜덤 초기화한다.

> **참고 — 32개 query가 만드는 information bottleneck**: Q-Former의 출력은 $32 \times 768$ 크기로 고정된다. ViT-L/14의 이미지 feature가 $257 \times 1024$인 것과 비교하면 훨씬 작다. 이 **의도적인 병목**이 BLIP-2 설계의 핵심이다. query들은 제한된 용량 안에 담아야 하므로 **언어 모델에게 가장 유용한 정보만 골라내도록** 강제되며, 불필요한 시각 정보는 자연히 걸러진다. 입력 이미지의 해상도나 patch 수와 무관하게 LLM이 받는 토큰 수가 항상 32개로 고정된다는 점도 연산 측면의 이점이다.

<br/>

### 1단계: Vision-Language Representation Learning

<!--Figure: Figure 2 (Self-attention masking strategies for each objective)-->

1단계에서는 Q-Former를 **frozen 이미지 인코더에만 연결**하여, image-text 쌍으로 학습한다. 목표는 query들이 **텍스트와 가장 관련이 깊은 시각 표현을 추출하도록** 만드는 것이다.

BLIP과 동일하게 세 가지 목적함수를 함께 최적화하는데, BLIP-2의 특징은 **세 목적함수가 동일한 입력 형식과 파라미터를 공유하면서, 오직 attention mask만 다르게 적용해 구분된다**는 점이다.

- **ITC (Image-Text Contrastive Learning)**: 이미지 표현과 텍스트 표현을 정렬한다. query 출력이 32개이므로 텍스트의 `[CLS]` 토큰과 각각 유사도를 계산한 뒤, **그중 가장 높은 값**을 해당 쌍의 유사도로 사용한다. 이때 **unimodal self-attention mask**를 적용해 query와 텍스트가 서로를 보지 못하게 하는데, 정답 텍스트를 미리 참조해버리는 **information leak**을 막기 위함이다.
- **ITG (Image-grounded Text Generation)**: 이미지를 조건으로 텍스트를 생성한다. Q-Former 구조상 **텍스트 토큰은 frozen 이미지 feature에 직접 접근할 수 없으므로**, 생성에 필요한 정보는 반드시 query가 먼저 추출한 뒤 self-attention을 통해 텍스트로 전달되어야 한다. **multimodal causal self-attention mask**를 사용해 query끼리는 서로 보되 텍스트는 보지 못하게 하고, 텍스트는 모든 query와 자신의 이전 토큰까지만 보게 한다. 디코딩 태스크임을 알리기 위해 `[CLS]` 대신 `[DEC]` 토큰을 첫 토큰으로 쓴다.
- **ITM (Image-Text Matching)**: 쌍이 실제로 맞는지 판별하는 이진 분류로 세밀한 정렬을 학습한다. **bi-directional self-attention mask**를 써서 query와 텍스트가 서로 자유롭게 attend하게 한 뒤, 각 query 출력을 2-class 분류기에 통과시켜 **logit을 평균**낸다. BLIP과 마찬가지로 hard negative mining을 적용한다.

> **참고 — ITG가 retrieval에도 도움이 되는 이유**: ITG는 생성 태스크지만 retrieval 성능도 함께 끌어올린다. 텍스트를 생성해내려면 query가 **텍스트에 담긴 정보를 빠짐없이 포착**해야 하기 때문이다. 정렬만 학습하는 ITC보다 더 완전한 시각 표현을 강제하는 셈이다.

또한 이미지 인코더가 frozen이라 표현이 안정적이므로, BLIP이 사용하던 momentum encoder와 queue 없이 **배치 내 negative만으로** ITC를 학습할 수 있다.

<br/>

### 2단계: Vision-to-Language Generative Learning

<!--Figure: Figure 3 (BLIP-2's second-stage generative pre-training)-->

2단계에서는 1단계로 학습된 Q-Former를 **frozen LLM에 연결**한다. 연결 방식은 매우 단순하다.

**하나의 fully-connected 레이어**가 query 출력 임베딩을 LLM의 텍스트 임베딩과 같은 차원으로 사영하고, 이렇게 사영된 32개 임베딩을 **입력 텍스트 임베딩 앞에 붙인다.** 즉 query 출력이 **soft visual prompt** 역할을 하여 LLM을 시각 정보에 조건화한다.

여기서 1단계의 의미가 드러난다. Q-Former는 이미 **언어와 정렬된 표현을 추출하도록 학습된 상태**이므로, LLM이 받는 것은 날것의 시각 feature가 아니라 **언어로 해석 가능한 형태로 정제된 정보**이다. LLM은 vision-language 정렬을 스스로 학습할 부담을 덜고, 그 덕분에 **catastrophic forgetting도 완화**된다.

LLM 종류에 따라 학습 방식이 갈린다.

- **Decoder 기반 LLM**(OPT): 일반적인 **language modeling loss**로 학습한다. LLM이 시각 표현을 조건으로 텍스트를 생성한다.
- **Encoder-Decoder 기반 LLM**(FlanT5): **prefix language modeling loss**로 학습한다. 텍스트를 둘로 나눠, 앞부분은 시각 표현과 함께 encoder 입력으로 넣고 뒷부분을 decoder의 생성 목표로 삼는다.

<br/>

### 학습 설정

사전 학습 데이터는 BLIP과 동일한 **1억 2900만 장** 규모로, COCO, Visual Genome, CC3M, CC12M, SBU에 LAION400M에서 가져온 1억 1500만 장을 더해 구성한다. 웹 이미지에는 BLIP의 **CapFilt를 적용**하는데, BLIP-large captioner로 nucleus sampling을 통해 캡션 10개를 생성한 뒤 **CLIP ViT-L/14의 image-text 유사도로 순위를 매겨 상위 2개만** 남기고, 학습 스텝마다 그중 하나를 무작위로 뽑아 쓴다.

이미지 인코더로는 **CLIP의 ViT-L/14**와 **EVA-CLIP의 ViT-g/14**를 사용한다. 이때 ViT의 **마지막 레이어를 제거하고 뒤에서 두 번째 레이어의 출력**을 쓰는 것이 성능이 조금 더 좋았다. LLM으로는 OPT 계열과 FlanT5 계열을 사용하며, frozen 모델들의 파라미터는 **FP16**(FlanT5는 BFloat16)으로 변환해도 성능 저하가 없었다.

학습 비용은 이 접근의 이점을 잘 보여준다. 가장 큰 모델인 **ViT-g + FlanT5-XXL** 조합조차 **A100(40G) 16장 한 대에서 1단계 6일 미만, 2단계 3일 미만**이면 학습이 끝난다.

<br/>

## 실험 결과

### 학습 효율과 zero-shot 성능

<!--Figure: Table 1 (Overview of BLIP-2 results on various zero-shot vision-language tasks)-->

- **압도적인 파라미터 효율**: BLIP-2는 zero-shot VQAv2에서 **Flamingo80B를 8.7% 앞서면서도 학습 가능한 파라미터는 54배 적다.** 기존 방법들이 수십억 개의 파라미터를 학습시킨 것과 달리, BLIP-2는 Q-Former와 FC 레이어만 학습한다.
- **다양한 태스크에서의 SOTA**: VQA, image captioning, image-text retrieval 전반에서 당시 최고 수준을 기록했다. 특히 **NoCaps zero-shot transfer**에서 강한 일반화 성능을 보였다.
- **Emerging zero-shot instructed generation**: 별도의 instruction tuning 없이도 자연어 지시를 따르는 능력이 나타났다. 시각적 지식 추론, 시각적 상식 추론, 시각 대화, 개인화된 image-to-text 생성 등이 프롬프트만으로 가능했다. **frozen LLM이 이미 갖고 있던 언어 능력이 시각 입력으로 확장된 결과**이다.

<br/>

### 1단계 표현 학습의 필수성

<!--Figure: Figure 5 (Effect of vision-language representation learning on generative learning)-->

논문에서 가장 중요한 ablation은 **1단계를 건너뛰면 어떻게 되는가**이다.

1단계 없이 Q-Former를 곧바로 LLM에 연결해 학습시키면, 두 종류의 LLM 모두에서 zero-shot VQA 성능이 **큰 폭으로 떨어졌다.** 특히 OPT에서는 학습이 진행될수록 정확도가 급격히 하락하는 **catastrophic forgetting**이 뚜렷하게 관찰되었다.

이는 두 단계의 역할 분담이 설계상의 편의가 아니라 **필수 조건**임을 보여준다. query가 언어와 정렬되지 않은 상태에서 LLM에 들어가면, LLM은 의미를 알 수 없는 벡터를 해석하느라 자신의 언어 능력을 훼손하게 된다. **정렬이 먼저, 생성 연결은 그다음**이라는 순서가 지켜져야 한다.

<br/>

### Image-Text Retrieval

retrieval은 생성을 필요로 하지 않으므로, **LLM 없이 1단계 모델만 직접 finetuning**한다. COCO에서 Q-Former와 이미지 인코더를 ITC, ITM, ITG로 함께 학습시켜 COCO와 Flickr30K에서 SOTA를 달성했다.

ablation에서 **ITG loss를 제거하면 retrieval 성능이 떨어졌는데**, 앞서 설명한 대로 생성 목적함수가 query로 하여금 텍스트의 정보를 더 온전히 포착하도록 강제하기 때문이다.

<br/>

## 한계점

- **In-context learning이 작동하지 않음**: 논문이 명시적으로 인정하는 가장 큰 한계이다. VQA 예시를 여러 개 프롬프트에 넣어줘도 **성능이 개선되지 않았다.** 원인은 데이터에 있다. 사전 학습 데이터가 **샘플당 image-text 쌍을 하나씩만** 담고 있어서, LLM이 **하나의 시퀀스 안에 여러 image-text 쌍이 놓인 상황의 상관관계를 학습할 기회가 없었다.** Flamingo가 few-shot에 강했던 것은 여러 쌍이 교차 배열된 비공개 데이터셋(M3W)을 썼기 때문이다.
- **LLM의 결함을 그대로 물려받음**: frozen LLM을 쓰는 대가로, LLM의 부정확한 지식, 잘못된 추론 경로, 최신 정보 부재가 그대로 출력에 반영된다. 또한 공격적인 표현 생성, 사회적 편향 전파, 개인정보 유출 같은 LLM의 위험도 함께 상속된다.
- **Frozen 구조 자체의 제약**: 이미지 인코더가 frozen이므로, 사전 학습된 인코더가 애초에 포착하지 못하는 시각 정보(예: 세밀한 텍스트 인식, 고해상도 디테일)는 Q-Former가 아무리 학습되어도 복원할 수 없다. **성능의 상한이 frozen 모델의 능력에 묶인다.**
- **32개 query라는 고정 병목**: 병목은 효율의 원천이자 동시에 제약이다. 이미지 하나를 32개 토큰으로 압축하므로, 조밀한 정보가 필요한 태스크(문서 이해, 다중 객체의 정밀한 위치 관계 등)에서는 정보 손실이 발생할 수 있다.
- **2단계 학습 파이프라인의 복잡성**: 서로 다른 목적함수와 attention mask를 갖는 두 단계를 순차적으로 거쳐야 하며, 재현과 튜닝의 난이도가 단순한 구조에 비해 높다. 이후 연구들이 더 단순한 연결 방식을 시도하게 되는 배경이기도 하다.

<br/>

## 인사이트

- **모듈 조립이라는 패러다임의 정착**: BLIP-2의 진짜 기여는 특정 아키텍처가 아니라 **"각 모달리티의 전문가 모델을 얼어붙인 채 가져다 쓰고, 그 사이만 학습한다"** 는 구성 방식 자체이다. 이 관점은 이후 **vision encoder + projector + LLM**이라는 멀티모달 모델의 표준 템플릿으로 굳어졌고, 오늘날 대부분의 VLM이 이 골격을 따른다.
- **정렬은 연결보다 먼저 와야 한다**: 1단계를 생략했을 때 나타난 catastrophic forgetting은, 모달리티를 잇는 작업에서 **순서가 성능을 좌우한다**는 것을 보여준다. 해석 가능한 형태로 표현을 먼저 정렬해두지 않으면, 강력한 LLM일수록 오히려 더 크게 망가진다.
- **병목이 곧 설계 도구**: 정보를 더 많이 전달하는 것이 항상 좋은 것은 아니다. 32개라는 고정 크기 제약이 오히려 **무엇이 중요한지를 학습하도록 강제하는 장치**로 작동했다. 표현 학습에서 용량 제한을 의도적인 설계 수단으로 쓴 사례이다.
- **연결자(connector) 설계를 둘러싼 논쟁의 출발점**: Q-Former는 정교하지만 복잡하다. 비슷한 시기의 **LLaVA**는 단순한 linear projection만으로도 좋은 성능을 냈고, LLaVA-1.5는 MLP와 충분한 instruction data가 Q-Former의 복잡성을 상당 부분 대체할 수 있음을 보였다. 다만 Q-Former의 **고정된 토큰 수**는 연산 효율 측면에서 여전히 뚜렷한 장점이라, 두 방향은 지금도 공존하며 트레이드오프를 이룬다.
- **연구 접근성의 확대**: A100 16장으로 9일이면 최대 모델을 학습할 수 있다는 것은, 대규모 멀티모달 연구가 소수의 기관에만 가능하던 상황을 바꾸는 의미를 갖는다. **frozen 모델 재사용이 곧 연구 민주화의 수단**이 된 사례이다.
- **한계가 곧 다음 연구의 출발점**: instruction을 제대로 따르지 못하는 문제는 instruction-aware Q-Former를 도입한 [InstructBLIP](instructblip.md)으로, in-context learning 부재는 interleaved 데이터를 활용하는 연구들로 이어졌다. BLIP이 "데이터를 어떻게 개선할 것인가"를 물었다면, BLIP-2는 **"이미 학습된 것을 어떻게 연결할 것인가"** 를 물었고, 그 답이 이후 VLM 설계의 기본형이 되었다.

<br/>
