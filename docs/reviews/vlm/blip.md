# BLIP

:::{admonition} 논문 정보
:class: note

- **제목**: BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation
- **저자**: Li et al.
- **기관**: Salesforce
- **발표**: ICML 2022
- **arXiv**: [링크](https://arxiv.org/abs/2201.12086)
:::

## 핵심 아이디어

BLIP은 **모델과 데이터 양쪽의 한계를 동시에 공략**한다. 하나는 이해(understanding)와 생성(generation)을 하나의 모델로 통합하는 것이고, 다른 하나는 **모델이 스스로 학습 데이터를 개선하도록 만드는 것**이다.

**모델 관점의 문제**는 기존 구조들이 한쪽 능력만 갖는다는 점이다.
* **Encoder 기반 모델**(CLIP, ALBEF)은 이미지-텍스트 정렬에는 강하지만 텍스트를 생성할 수 없어 image captioning 같은 태스크에 직접 쓸 수 없다.
* **Encoder-Decoder 기반 모델**(SimVLM)은 생성이 가능하지만, 이미지-텍스트 retrieval에는 성공적으로 적용되지 못했다.

**데이터 관점의 문제**는 더 근본적이다. CLIP, ALIGN, SimVLM 등 당시의 SOTA는 모두 웹에서 수집한 **noisy한 alt-text**를 대규모로 긁어모아 성능을 끌어올렸다. 규모를 키우면 성능이 오르는 것은 분명하지만, 웹 텍스트의 상당수는 이미지를 제대로 설명하지 않는다. 즉 **noisy한 supervision은 vision-language 학습에 차선책일 뿐**인데, 기존 연구들은 이를 감수해야 할 비용으로 받아들였다.

BLIP은 이 데이터 문제를 **모델을 이용해 해결한다**. 사전 학습된 모델로부터 **캡션을 생성하는 Captioner**와 **잘못된 캡션을 걸러내는 Filter**를 만들고, 이 둘로 웹 데이터를 정제한 뒤 그 데이터로 모델을 다시 학습시킨다. 데이터가 모델을 만들고, 그 모델이 다시 데이터를 개선하는 **부트스트랩 구조**이다.

## 방법론

BLIP의 방법론은 두 축으로 구성된다. 통합 아키텍처인 **MED**와, 데이터 부트스트랩 기법인 **CapFilt**이다.

### MED (Multimodal mixture of Encoder-Decoder)

<!--Figure: Figure 2 (Pre-training model architecture and objectives)-->

MED는 하나의 모델이 **세 가지 모드로 동작**할 수 있도록 설계된 구조이다. 이미지 인코더로는 ViT(ViT-B/16, ViT-L/16)를 사용하고, 텍스트 측이 다음 세 역할을 겸한다.

- **Unimodal Encoder**: 이미지와 텍스트를 각각 독립적으로 인코딩한다. 텍스트 측은 BERT와 동일한 구조이며, 문장 앞에 `[CLS]` 토큰을 붙여 문장 표현으로 사용한다.
- **Image-grounded Text Encoder**: 각 transformer 블록의 self-attention과 FFN 사이에 **cross-attention 레이어를 추가로 삽입**하여 이미지 정보를 텍스트에 주입한다. 텍스트 앞에 `[Encode]` 토큰을 붙이고, 그 출력 임베딩을 **image-text 쌍의 멀티모달 표현**으로 사용한다.
- **Image-grounded Text Decoder**: 위 구조에서 **bidirectional self-attention을 causal self-attention으로 교체**한 형태이다. 문장 시작을 알리는 `[Decode]` 토큰으로 생성을 시작하며, autoregressive하게 캡션을 생성한다.

<br/>

### 세 가지 사전 학습 목적함수

세 모드는 각각 대응하는 손실 함수로 **동시에 학습**된다.

- **ITC (Image-Text Contrastive Loss)** — Unimodal Encoder에 적용된다. 짝을 이루는 image-text의 표현은 가깝게, 그렇지 않은 쌍은 멀게 만들어 **두 모달리티의 feature space를 정렬**한다. ALBEF를 따라 momentum encoder와 soft label을 사용하여, 웹 데이터의 negative 중에 실제로는 정답에 가까운 샘플이 섞여 있는 문제를 완화한다.
- **ITM (Image-Text Matching Loss)** — Image-grounded Text Encoder에 적용된다. 주어진 image-text 쌍이 실제 짝인지 아닌지를 판별하는 **이진 분류**로, cross-attention을 통해 **세밀한 정렬(fine-grained alignment)** 을 학습한다. 이때 ITC 유사도가 높은(즉 구분하기 어려운) 샘플을 negative로 더 자주 뽑는 **hard negative mining**을 적용해 학습 신호를 강화한다.
- **LM (Language Modeling Loss)** — Image-grounded Text Decoder에 적용된다. 이미지가 주어졌을 때 캡션을 autoregressive하게 생성하도록 cross-entropy로 학습하며, label smoothing 0.1을 적용한다.

> **참고 — 왜 MLM이 아니라 LM인가**: 기존 VLP 연구들이 널리 쓰던 MLM(Masked Language Modeling)은 빈칸을 채우는 방식이라 **텍스트 생성 능력을 직접 학습하지 못한다.** BLIP은 이를 LM으로 대체함으로써, 사전 학습된 모델을 곧바로 captioner로 활용할 수 있게 만들었다. CapFilt가 성립하기 위한 전제 조건이기도 하다.

<br/>

### 파라미터 공유 전략

세 모드를 각각 별도 모델로 두면 파라미터가 3배로 늘어난다. BLIP은 텍스트 인코더와 디코더가 **self-attention 레이어를 제외한 모든 파라미터를 공유**한다.

이렇게 나눈 근거는 두 태스크의 본질적 차이가 **self-attention에 집중되어 있다**는 관찰이다. 인코딩은 현재 입력 전체의 표현을 만들기 위해 bidirectional attention을 써야 하고, 디코딩은 다음 토큰을 예측해야 하므로 미래를 가리는 causal attention을 써야 한다. 반면 embedding 레이어, cross-attention, FFN은 두 태스크에서 하는 역할이 유사하므로 공유해도 무방하다.

결과적으로 파라미터 효율을 얻으면서 **multi-task learning의 이점도 함께 누린다.** 다만 학습 시에는 image-text 쌍 하나당 무거운 ViT를 1회, 텍스트 transformer를 3회 forward해야 한다.

<br/>

### CapFilt (Captioning and Filtering)

<!--Figure: Figure 3 (Learning framework of BLIP with CapFilt)-->

CapFilt는 BLIP의 가장 독창적인 기여로, **사전 학습된 모델을 이용해 학습 데이터 자체를 개선**하는 절차이다.

먼저 데이터는 두 종류로 나뉜다. 웹에서 수집한 대량의 noisy 쌍 $(I_w, T_w)$와, 사람이 직접 작성한 소량의 고품질 쌍 $(I_h, T_h)$(COCO)이다.

절차는 다음과 같다.

1. **사전 학습**: 전체 데이터로 MED를 사전 학습한다.
2. **Captioner와 Filter 초기화**: 사전 학습된 MED로부터 두 모듈을 만들고, **각각 COCO에서 개별적으로 finetuning**한다.
   - **Captioner**는 image-grounded text decoder이며 LM 목적함수로 학습된다. 웹 이미지 $I_w$에 대해 **합성 캡션 $T_s$** 를 생성한다.
   - **Filter**는 image-grounded text encoder이며 ITC와 ITM 목적함수로 학습된다.
3. **필터링**: Filter가 ITM 헤드로 판단하여 이미지와 맞지 않는 텍스트를 제거한다. 중요한 점은 **원본 웹 텍스트 $T_w$와 합성 캡션 $T_s$ 양쪽 모두를 필터링 대상으로 삼는다**는 것이다.
4. **재학습**: 필터를 통과한 쌍들과 사람이 작성한 쌍을 합쳐 새로운 데이터셋을 구성하고, 이것으로 **모델을 처음부터 다시 사전 학습**한다.

> **참고 — Captioner와 Filter를 왜 따로 두는가**: 두 모듈이 파라미터를 공유하면 성능이 오히려 떨어진다. Captioner가 만들어낸 잘못된 캡션을, 같은 파라미터를 쓰는 Filter가 걸러내지 못하는 **확증 편향(confirmation bias)** 이 발생하기 때문이다. 생성하는 쪽과 검증하는 쪽을 독립적으로 두는 것이 부트스트랩이 작동하기 위한 조건이다.

<br/>

사전 학습 데이터는 ALBEF와 동일하게 **총 1400만 장** 규모로, 사람이 작성한 COCO와 Visual Genome, 웹에서 수집한 Conceptual Captions(CC3M), Conceptual 12M, SBU Captions로 구성된다. 추가로 **LAION에서 1억 1500만 장**을 더한 대규모 설정도 실험한다.

<br/>

## 실험 결과

### CapFilt의 효과

<!--Figure: Table 1 (Effect of the captioner and filter)-->

- **Captioner와 Filter는 각각 독립적으로 성능을 개선**하며, 둘을 함께 적용했을 때 가장 큰 향상을 얻는다. 이는 두 모듈이 서로 다른 방식으로 데이터 품질에 기여함을 보여준다. Captioner는 **부족한 정보를 채워 넣고**, Filter는 **잘못된 정보를 제거한다.**
- **규모가 커져도 효과가 유지된다**: 데이터를 1400만 장에서 1억 2900만 장으로, 모델을 ViT-B에서 ViT-L로 키운 설정에서도 CapFilt의 이득이 그대로 나타났다. 데이터 부트스트랩이 단순히 소규모 설정에서만 통하는 트릭이 아니라 **scalable한 개선 방향**임을 뒷받침한다.

<br/>

### 캡션 생성 방식: Nucleus Sampling vs Beam Search

<!--Figure: Table 2 (Comparison between beam search and nucleus sampling)-->

합성 캡션을 어떤 디코딩 방식으로 생성할지에 대한 ablation은 직관에 반하는 결과를 보여준다.

- **Beam search**는 확률이 가장 높은 안전한 문장을 만들어내므로 노이즈 비율이 낮지만(필터 기준 약 19%), **데이터셋에 흔한 표현을 반복**하여 모델이 새로 배울 정보가 적다.
- **Nucleus sampling**($p=0.9$)은 확률적으로 샘플링하므로 노이즈 비율이 더 높지만(약 25%), **더 다양하고 예상 밖의 표현**을 만들어낸다. 결과적으로 downstream 성능은 nucleus sampling 쪽이 더 좋았다.

즉 합성 데이터의 가치는 **정확성보다 새로운 정보량(diversity)** 에서 나오며, 노이즈는 Filter가 사후적으로 처리하면 된다는 것이 BLIP의 관점이다.

<br/>

### Downstream Task 성능

BLIP은 이해와 생성을 아우르는 광범위한 태스크에서 당시 SOTA를 달성했다. 논문이 제시하는 대표적인 개선 폭은 다음과 같다.

- **Image-Text Retrieval**: COCO에서 average recall@1 기준 **+2.7%**
- **Image Captioning**: COCO에서 CIDEr 기준 **+2.8%**
- **VQA**: VQA score 기준 **+1.6%**

이외에도 NLVR2(visual reasoning), VisDial(visual dialog)에서도 경쟁력 있는 성능을 보였다. 주목할 점은 이 성능들이 **동일한 사전 학습 모델 하나에서 파생**되었다는 것으로, MED의 통합 설계가 실제로 작동함을 뒷받침한다.

<br/>

### 비디오-언어 태스크로의 Zero-shot 전이

BLIP은 이미지로만 학습되었음에도, 비디오 프레임을 샘플링해 입력하는 단순한 방식으로 비디오 태스크에 직접 적용할 수 있다. **MSRVTT text-to-video retrieval**에서 zero-shot BLIP은 **비디오 데이터로 직접 finetuning된 기존 모델들을 능가**했다.

이는 BLIP이 학습한 vision-language 표현이 특정 도메인에 과적합되지 않은 **범용적인 표현**임을 보여주는 결과이다.

<br/>

## 한계점

- **추가 사전 학습 라운드로 인한 연산 비용**: CapFilt는 부트스트랩된 데이터로 모델을 **처음부터 다시 사전 학습**해야 하므로, 전체 파이프라인의 연산 비용이 사실상 두 배가 된다. 성능 향상이 순수하게 데이터 품질 개선에서 온 것인지, 추가 학습 자체의 효과가 섞인 것인지 분리하기 어렵다는 지적도 가능하다.
- **COCO 의존성과 스타일 편향**: Captioner와 Filter는 모두 사람이 작성한 COCO 데이터로 finetuning된다. 따라서 생성되는 합성 캡션은 **COCO의 짧고 객체 중심적인 문체와 도메인 특성을 그대로 물려받으며**, 웹 이미지의 다양한 맥락을 충분히 표현하지 못할 수 있다. 결국 부트스트랩의 상한이 소량의 human annotation에 의해 결정된다.
- **이미지당 하나의 캡션**: 생성 모델을 갖추고 있음에도 이미지당 합성 캡션을 하나만 만들어 사용한다. 논문 스스로 여러 개의 캡션을 생성해 코퍼스를 더 키우는 방향을 후속 과제로 제시하며, 생성 능력이 충분히 활용되지 않았음을 인정한다.
- **부트스트랩 반복의 미검증**: CapFilt는 단 한 번의 라운드만 수행되었다. 여러 라운드를 반복하거나 여러 개의 captioner·filter를 앙상블하는 방향은 향후 과제로 남겨져 있어, **부트스트랩이 어디까지 누적될 수 있는지**는 확인되지 않았다.
- **학습 시 연산 오버헤드**: 세 가지 목적함수를 동시에 학습하므로, 쌍 하나당 텍스트 transformer를 **세 번 forward**해야 한다. 파라미터는 공유로 절약했지만 연산량은 그만큼 늘어난다.
- **Filter의 판단 기준이 이진적**: 노이즈 여부를 ITM 헤드의 binary 판정으로만 결정하므로, **부분적으로만 맞는 캡션**(일부는 정확하고 일부는 틀린 경우)을 다루지 못하고 통째로 버리거나 통째로 남긴다. 캡션의 hallucination을 세밀하게 교정하는 메커니즘은 없다.

<br/>

## 인사이트

- **데이터를 고정된 자산이 아니라 개선 대상으로 본 전환**: 그전까지 대규모 사전 학습 연구의 암묵적 전제는 "웹 데이터는 noisy하지만 많으니 그대로 쓴다"였다. BLIP은 **모델을 데이터 파이프라인 안에 집어넣어**(model-in-the-loop) 그 전제를 깼다. 이 관점은 이후 LLaVA의 GPT-4 기반 instruction data 생성, LAION-COCO의 합성 캡션 재라벨링 등 **synthetic data 흐름 전반의 초기 사례**가 되었다.
- **생성 능력이 곧 데이터 생산 수단이 된다**: MLM 대신 LM을 목적함수로 채택한 선택은 단순히 captioning 태스크를 풀기 위한 것이 아니었다. 생성할 수 있는 모델이어야 스스로 데이터를 만들 수 있고, 그 데이터로 다시 더 나은 모델을 만들 수 있다. **아키텍처 선택이 데이터 전략을 가능하게 한 구조**로, 이후 self-improvement 계열 연구가 공유하는 논리이다.
- **생성자와 검증자의 분리**: Captioner와 Filter가 파라미터를 공유하면 확증 편향으로 성능이 떨어진다는 관찰은, 합성 데이터를 다루는 모든 파이프라인에 적용되는 일반적 교훈이다. **만드는 쪽과 거르는 쪽이 독립적이어야 품질 개선이 실제로 일어난다**는 원리는 이후 LLM의 self-critique, reward model, LLM-as-a-judge 설계와도 통한다.
- **다양성이 정확성보다 유용할 수 있다**: nucleus sampling이 beam search보다 노이즈가 많은데도 더 좋은 결과를 낸 것은, 학습 데이터의 가치가 **개별 샘플의 정확도가 아니라 코퍼스 전체가 담은 정보량**에 있음을 보여준다. 노이즈는 필터로 사후 처리하고 다양성은 생성 단계에서 확보한다는 역할 분담이 핵심이다.
- **이해와 생성의 통합, 그리고 BLIP-2로의 연결**: CLIP이 dual-encoder로 정렬에만 집중했다면 BLIP은 ITC·ITM·LM을 결합해 하나의 모델로 둘 다 해냈다. 다만 모든 것을 처음부터 학습해야 한다는 비용 문제가 남았고, 이는 **frozen 이미지 인코더와 frozen LLM을 Q-Former로 연결**하는 [BLIP-2](blip2.md)로 이어진다. BLIP이 "무엇을 학습시킬 데이터인가"를 물었다면, BLIP-2는 "이미 학습된 것을 어떻게 연결할 것인가"를 묻는 방향 전환이다.

<br/>
