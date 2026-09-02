# InstructBLIP

:::{admonition} 논문 정보
:class: note

- **제목**: InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning
- **저자**: Dai et al.
- **기관**: Salesforce / HKUST / NTU
- **발표**: NeurIPS 2023
- **arXiv**: [링크](https://arxiv.org/abs/2305.06500)
:::

## 핵심 아이디어

InstructBLIP은 [BLIP-2](blip2.md)의 사전 학습 체크포인트 위에서 출발해, **vision-language instruction tuning을 체계적으로 파고든 연구**이다. 핵심 질문은 두 가지이다. **어떤 데이터로 instruction tuning을 해야 하는가**, 그리고 **시각 특징 추출 자체가 instruction을 알아야 하지 않는가**이다.

두 번째 질문이 이 논문의 가장 독창적인 지점이다.

BLIP-2의 Q-Former는 이미지로부터 32개의 query 토큰을 뽑아내는데, 이때 **사용자가 무엇을 물었는지와 무관하게 항상 같은 특징을 추출한다.** 같은 사진을 두고 "이 사람이 입은 옷의 색은?"과 "배경에 보이는 건물은 몇 층인가?"를 물어도, LLM에 전달되는 32개 토큰은 완전히 동일하다. 질문에 답하는 일은 전적으로 LLM의 몫이 된다.

그런데 32개 토큰은 [앞서 살펴본 대로](blip2.md) **의도적으로 좁혀놓은 병목**이다. 질문과 무관하게 압축한다면, 정작 그 질문에 필요한 정보가 병목을 통과하지 못하고 버려질 수 있다.

InstructBLIP의 답은 단순하다. **instruction 텍스트를 Q-Former에도 함께 넣어주는 것**이다. 그러면 query들이 "지금 무엇을 물어보고 있는지"를 알고 이미지를 바라보게 되고, **태스크에 필요한 시각 특징을 선택적으로 추출**할 수 있다. 이것이 **instruction-aware Q-Former**이다.

여기에 **26개 공개 데이터셋을 instruction 형식으로 변환**하고, 크기가 제각각인 데이터셋들을 균형 있게 섞는 sampling 전략을 더해, 13개 held-out 데이터셋 전부에서 zero-shot SOTA를 달성한다.

## 방법론

### Instruction-aware Q-Former

<!--Figure: Figure 2 (Model architecture of InstructBLIP)-->

구조적 변경은 한 지점에 집중되어 있다. **instruction 토큰이 Q-Former의 입력에 추가된다.**

동작 순서는 다음과 같다.

1. instruction 토큰과 학습 가능한 query 임베딩이 Q-Former의 **self-attention 레이어에서 상호작용**한다. 이 과정에서 query들이 instruction의 의도를 흡수한다.
2. 의도를 갖게 된 query들이 **cross-attention을 통해 frozen 이미지 feature를 참조**한다.
3. 결과적으로 **instruction에 맞춰 선별된 시각 표현**이 만들어진다.

중요한 점은 **instruction이 Q-Former와 frozen LLM 양쪽 모두에 주어진다**는 것이다. LLM은 여전히 질문을 직접 보고 답하되, 그 앞단에서 받는 시각 정보가 이미 질문에 맞게 정제되어 있는 셈이다.

> **참고 — 무엇이 바뀐 것인가**: BLIP-2에서 시각 인코딩은 질문과 **독립적인 전처리**였다. 이미지를 한 번 인코딩해두면 어떤 질문에도 같은 표현을 쓴다. InstructBLIP은 이를 **질문에 의존하는 과정**으로 바꿨다. "이미지를 인코딩한다"에서 **"이 질문을 위해 이미지를 인코딩한다"** 로의 전환이며, 고정 크기 병목을 쓰는 구조에서는 특히 큰 차이를 만든다.

<br/>

### Instruction Tuning 데이터 구성

<!--Figure: Figure 2 (Tasks and datasets used for training and evaluation)-->

논문은 **26개의 공개 데이터셋**을 수집해 instruction tuning 형식으로 변환했다. 다루는 태스크 범주는 image captioning, 글자 읽기가 필요한 captioning, VQA, 지식 기반 image QA, image QA 생성, video QA, visual reasoning, visual conversational QA, image classification, 그리고 LLaVA-Instruct-150K까지 폭넓다.

각 태스크마다 **10~15개의 서로 다른 자연어 instruction 템플릿**을 직접 작성했다. 표현이 하나로 고정되면 모델이 문구 자체에 과적합되기 때문이다.

> **참고 — 짧은 답변 편향에 대한 대응**: 수집한 공개 데이터셋 상당수는 단답형 정답을 갖는 학술 벤치마크이다. 그대로 학습시키면 모델이 **무조건 짧게 답하는 습관**을 들이게 된다. 논문은 일부 템플릿에 "short", "briefly" 같은 단어를 명시적으로 넣어, **짧은 답이 요구된 경우에만 짧게 답하도록** 조건을 학습시키는 방식으로 이를 완화했다.

**held-in / held-out 분할**이 이 논문의 평가 방법론적 기여이다. 26개를 **13개 held-in(학습용)** 과 **13개 held-out(zero-shot 평가용)** 으로 나누되, held-out을 두 수준으로 구분했다.

- **held-out 데이터셋**: 해당 데이터셋은 못 봤지만, **같은 종류의 태스크는 학습 중에 접한** 경우
- **held-out 태스크**: video QA, visual conversational QA, image classification, visual reasoning처럼 **태스크 범주 자체를 통째로 학습에서 제외**한 경우

두 수준을 나눔으로써 "새로운 데이터에 대한 일반화"와 "새로운 종류의 문제에 대한 일반화"를 구분해 측정할 수 있게 된다.

<br/>

### Balanced Dataset Sampling

26개 데이터셋은 크기가 제각각이다. 이를 그대로 균등하게 섞으면 **작은 데이터셋에는 과적합되고 큰 데이터셋은 충분히 학습되지 않는다.**

InstructBLIP은 각 데이터셋을 **크기의 제곱근에 비례하는 확률**로 샘플링한다.

$$
p_d = \frac{\sqrt{S_d}}{\sum_{i} \sqrt{S_i}}
$$

여기서 $S_d$는 데이터셋 $d$의 학습 샘플 수이다. 제곱근을 취하면 큰 데이터셋의 지배력이 완화되면서도 크기 차이는 반영된다. 여기에 태스크 특성을 고려한 수동 조정을 더했는데, 예를 들어 객관식인 A-OKVQA는 가중치를 낮추고 OKVQA는 높였다.

<br/>

### 학습 설정

- **Frozen 이미지 인코더**: EVA-CLIP의 **ViT-g/14** (BLIP-2와 동일하게 마지막 레이어를 제거하고 뒤에서 두 번째 출력 사용)
- **Frozen LLM**: FlanT5-XL(3B), FlanT5-XXL(11B), Vicuna-7B, Vicuna-13B의 네 가지
- **초기화**: 2단계 사전 학습을 마친 **BLIP-2 체크포인트에서 Q-Former를 가져온다**
- **학습 대상**: instruction tuning 과정에서 **오직 Q-Former만 학습**되며, 이미지 인코더와 LLM은 계속 frozen이다

즉 InstructBLIP은 BLIP-2의 frozen 철학을 그대로 유지한 채, **학습되는 유일한 모듈에 instruction을 알려주는** 변경만 가한 것이다.

<br/>

## 실험 결과

### Zero-shot 성능

<!--Figure: Table 1 (Zero-shot results on the held-out datasets)-->

- **13개 held-out 데이터셋 전부에서 SOTA**를 달성했다. 일부가 아니라 전부라는 점이 강조된다.
- BLIP-2와 Flamingo를 큰 폭으로 앞섰다. 특히 **InstructBLIP FlanT5-XL(4B)이 Flamingo-80B를 평균 24.8% 능가**했는데, 모델 크기가 20분의 1임을 감안하면 **instruction tuning의 효과가 규모의 차이를 뒤집을 수 있음**을 보여준다.
- **held-out 태스크**(video QA 등 범주 자체를 못 본 경우)에서도 일반화가 나타났다. 이미지로만 학습했음에도 video QA에서 경쟁력 있는 성능을 보였다.

<br/>

### Finetuning 성능

InstructBLIP 체크포인트는 개별 downstream 태스크로 finetuning할 때 **더 좋은 출발점**이 되었다. 대표적으로 이미지가 포함된 **ScienceQA 문항에서 90.7%** 를 기록했다.

instruction tuning이 zero-shot 능력만 올리는 것이 아니라, **이후의 특화 학습에도 유리한 표현을 남긴다**는 점을 보여주는 결과이다.

<br/>

### Ablation

논문의 두 가지 핵심 설계가 모두 실제로 기여함이 확인되었다.

- **Instruction-aware 시각 특징을 제거하면 성능이 크게 떨어졌다.** 특히 **공간적·시간적 추론을 요구하는 태스크**에서 하락 폭이 컸다. ScienceQA의 이미지 문맥 문항과 iVQA 같은 비디오 태스크가 대표적이다. 이미지의 어느 부분을, 혹은 어느 시점을 봐야 하는지가 질문에 따라 달라지는 태스크일수록 **질문을 아는 것이 중요**하다는 해석이 가능하다.
- **Balanced sampling을 제거하면 성능이 떨어지고 학습이 불안정해졌다.** 데이터셋 크기 불균형이 단순한 성능 문제가 아니라 **학습 안정성의 문제**이기도 함을 보여준다.

<br/>

## 한계점

- **단답형 편향**: 학습 데이터의 대부분이 짧은 정답을 갖는 학술 벤치마크이다. 템플릿에 "briefly" 같은 단서를 넣어 완화했지만, **긴 호흡의 개방형 생성이나 자유로운 대화 능력은 [LLaVA](llava.md) 계열에 비해 약하다.** 벤치마크 점수와 실제 사용 경험이 갈리는 지점이다.
- **실사용 instruction과의 분포 차이**: 26개 데이터셋은 연구 목적으로 구축된 것이라, 실제 사용자가 던지는 자유로운 요청과는 분포가 다르다. **벤치마크 커버리지가 넓다는 것이 실사용 일반성을 보장하지는 않는다.**
- **Q-Former만 학습한다는 제약**: 파라미터 효율은 뛰어나지만, 적응할 수 있는 용량 자체가 Q-Former로 제한된다. LLM이 애초에 갖지 못한 능력은 instruction tuning으로 끌어낼 수 없다.
- **BLIP-2의 한계를 그대로 상속**: 224 해상도의 frozen 인코더에서 오는 정밀 인식의 한계, 다중 이미지 미지원, 그리고 **in-context learning 부재**가 그대로 남아 있다. 학습 샘플이 여전히 이미지 하나에 텍스트 하나 구조이므로, 예시를 여러 개 넣어주는 few-shot 프롬프팅은 작동하지 않는다.
- **held-out 분할의 엄밀성**: held-out 데이터셋 중 일부는 held-in 데이터셋과 **이미지 출처를 공유**한다. 논문도 이를 인지해 일부를 제외했지만, "본 적 없다"는 기준을 데이터셋 단위로 정의하는 방식 자체에 한계가 있다.
- **안전성 정렬의 부재**: instruction 수행 능력에 초점을 맞췄을 뿐, hallucination 억제나 유해 출력 방지를 위한 별도의 정렬 과정은 포함되지 않았다.

<br/>

## 인사이트

- **시각 인코딩은 질문에 의존해야 한다**: InstructBLIP의 핵심 통찰은 **"이미지를 인코딩한다"가 아니라 "이 질문을 위해 이미지를 인코딩한다"** 는 관점 전환이다. 이미지를 한 번 압축해두고 재사용하는 방식은 효율적이지만, 압축 과정에서 무엇을 버릴지 결정하려면 목적을 알아야 한다. **고정 크기 병목을 쓰는 모든 구조에 적용되는 원리**이며, 병목이 좁을수록 이 효과는 커진다.
- **두 가지 데이터 철학의 분기**: 비슷한 시기에 LLaVA는 GPT-4가 생성한 대화 데이터에, InstructBLIP은 26개 학술 데이터셋의 태스크 다양성에 베팅했다. 결과적으로 **LLaVA는 자유로운 대화에, InstructBLIP은 벤치마크 zero-shot에 강했다.** 어느 쪽이 옳았다기보다, **instruction tuning 데이터의 성격이 모델의 성격을 그대로 결정한다**는 사실이 드러난 것이다. 이후 Qwen-VL, InternVL 등은 두 종류를 모두 섞는 방향으로 수렴한다.
- **평가 방법론의 기여**: held-in / held-out을 나누고, held-out을 다시 "데이터셋 수준"과 "태스크 수준"으로 구분한 설계는 멀티모달 instruction tuning 평가에 **일반화의 층위를 구분하는 기준**을 도입했다. 지금은 당연해 보이지만, 당시 관행은 학습에 쓴 것과 겹치는 벤치마크로 평가하는 경우가 흔했다.
- **데이터 혼합은 그 자체로 하나의 기법**: 제곱근 비례 샘플링은 화려하지 않지만, 이를 빼면 성능뿐 아니라 **학습 안정성까지 무너졌다.** 크기가 다른 데이터를 섞는 문제는 부수적인 엔지니어링이 아니라 결과를 좌우하는 설계 요소이며, 이후 멀티태스크 학습의 표준 레시피가 되었다.
- **작은 모델이 큰 모델을 이기는 조건**: 4B 모델이 80B 모델을 평균 24.8% 앞섰다는 결과는, **적절한 instruction tuning이 스무 배의 규모 차이를 상쇄할 수 있음**을 보여준다. 다만 이는 Flamingo가 instruction tuning을 전혀 하지 않았기 때문이기도 하므로, 규모가 무의미하다는 뜻이 아니라 **정렬되지 않은 규모는 낭비된다**는 쪽에 가깝다.

<br/>
