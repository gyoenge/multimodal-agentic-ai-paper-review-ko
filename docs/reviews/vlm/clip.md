# CLIP

:::{admonition} 논문 정보
:class: note

- **제목**: Learning Transferable Visual Models From Natural Language Supervision
- **저자**: Radford et al.
- **기관**: OpenAI
- **발표**: ICML 2021
- **arXiv**: [링크](https://arxiv.org/abs/2103.00020)
:::

## 핵심 아이디어

CLIP은 **자연어를 supervision 신호로 사용**하여, 고정된 카테고리 집합에 얽매이지 않는 범용 visual representation을 학습한다.

기존의 컴퓨터 비전 모델은 ImageNet의 1000개 클래스처럼 **미리 정해진 고정된 label 집합**에 대해 학습된다. 이 방식은
* (i) 새로운 개념을 인식하려면 매번 추가적인 labeled data와 재학습이 필요하고,
* (ii) 데이터셋 구축 비용이 커서 학습 데이터의 규모와 다양성이 제한되며,
* (iii) 모델이 학습한 개념 외에는 아무것도 표현할 수 없다는

한계를 갖는다. 즉 supervision의 형태 자체가 모델의 일반성을 가로막는 병목이 된다.

CLIP은 이 병목을 **"이 이미지와 짝을 이루는 텍스트는 무엇인가"** 라는 문제로 대체한다. 인터넷에는 이미지와 그것을 설명하는 텍스트 쌍이 사실상 무한히 존재하므로, 사람이 일일이 라벨링하지 않고도 **4억 쌍 규모의 학습 데이터**를 확보할 수 있다.
더 중요한 것은 supervision이 자연어라는 점이다. 자연어는 고정된 클래스 인덱스와 달리 **임의의 개념을 조합해 표현할 수 있으므로**, 학습된 모델은 추론 시점에 텍스트만 바꿔주면 어떤 카테고리 집합에 대해서든 분류기로 동작할 수 있다.

그 결과 CLIP은 ImageNet의 학습 이미지 128만 장을 **단 한 장도 사용하지 않고도** zero-shot만으로 supervised ResNet-50과 동등한 정확도를 달성한다.

## 방법론

### Natural Language Supervision과 WIT 데이터셋

CLIP의 출발점은 학습 데이터의 규모와 형태이다.
기존의 image-text 연구들이 사용하던 MS-COCO, Visual Genome은 고품질이지만 각각 약 10만 장 규모에 불과하고, YFCC100M은 1억 장 규모이지만 메타데이터 품질이 매우 낮아 자연어 설명이 붙은 이미지만 추리면 1500만 장 수준으로 줄어든다.

이에 저자들은 인터넷의 공개 데이터로부터 **4억 개의 (image, text) 쌍**으로 구성된 **WIT(WebImageText)** 데이터셋을 새로 구축한다. 50만 개의 검색 쿼리를 설정하고, 쿼리당 최대 2만 쌍을 수집하여 클래스 균형을 맞추는 방식이다.
이 규모는 GPT-2의 학습 데이터인 WebText와 유사한 수준의 단어 수를 갖는다.

<br/>

### 학습 목표: Predictive에서 Contrastive로

<!--Figure: Figure 2 (data efficiency of contrastive objective)-->

초기 실험에서 저자들은 이미지로부터 **캡션의 정확한 단어열을 예측**하는 방식(VirTex와 유사한 image CNN + text transformer)을 시도했다. 그러나 이 방식은 학습 효율이 매우 낮았다. 하나의 이미지를 설명하는 방법은 무수히 많은데, 그중 **정확히 그 한 문장을 맞히도록** 강제하는 것은 지나치게 어려운 과제이기 때문이다.

논문은 목적함수를 두 단계에 걸쳐 완화하며 효율을 개선한다.

- Transformer 언어 모델로 캡션을 예측하는 방식 → 단어 순서를 무시하는 **bag-of-words 예측**으로 바꾸자 zero-shot 전이 효율이 **3배** 향상
- 여기서 다시 **contrastive 목적함수**로 바꾸자 추가로 **4배** 향상

즉 "무엇이 쓰여 있는지 정확히 생성"하는 대신 **"어떤 텍스트가 이 이미지와 짝인지 고르는"** 훨씬 쉬운 문제로 바꾼 것이, CLIP을 4억 쌍 규모로 학습 가능하게 만든 결정적 설계이다.

<br/>

### Contrastive Pre-training

<!--Figure: Figure 1 (a) Contrastive pre-training-->

$N$개의 (image, text) 쌍으로 이루어진 배치가 주어지면, CLIP은 가능한 $N \times N$개의 조합 중 **실제로 짝을 이루는 $N$개**를 맞히도록 학습한다.

이미지 인코더 $f_I$와 텍스트 인코더 $f_T$가 각각의 입력을 임베딩한 뒤, linear projection을 거쳐 동일한 multi-modal 임베딩 공간으로 사상하고 L2 정규화한다. 이렇게 얻은 이미지 임베딩 $\mathbf{I}_i$와 텍스트 임베딩 $\mathbf{T}_j$의 코사인 유사도에 학습 가능한 temperature $\tau$를 적용해 logit을 구성한다.

$$
\text{logits}_{ij} = \frac{\mathbf{I}_i \cdot \mathbf{T}_j}{\tau}
$$

손실은 이 유사도 행렬에 대해 **행 방향(image → text)과 열 방향(text → image)의 cross-entropy를 각각 계산한 뒤 평균**낸 symmetric 형태이다.

$$
\mathcal{L} = \frac{1}{2}\left( \mathcal{L}_{\text{image} \rightarrow \text{text}} + \mathcal{L}_{\text{text} \rightarrow \text{image}} \right)
$$

대각 성분($N$개의 정답 쌍)의 유사도는 높이고, 비대각 성분($N^2 - N$개의 오답 쌍)의 유사도는 낮추는 방향으로 두 인코더가 함께 학습된다.

> **참고 — Temperature $\tau$**: temperature는 softmax 분포의 날카로움을 조절하는 값으로, 값이 작을수록 유사도가 높은 쌍에 확률이 집중된다. CLIP은 이를 하이퍼파라미터로 고정하지 않고 **log-parameterized scalar로 두어 학습 과정에서 함께 최적화**하며, 학습 불안정을 막기 위해 scaling 값이 100을 넘지 않도록 clipping한다.

> **참고 — 배치 크기가 곧 난이도**: contrastive 학습에서 배치 내의 나머지 $N-1$개 샘플이 negative 역할을 하므로, **배치가 클수록 문제가 어려워지고 표현 학습의 신호도 강해진다.** CLIP이 32,768이라는 매우 큰 배치를 사용하는 이유이며, 이는 이후 대조 학습 기반 멀티모달 모델들이 공통적으로 따르는 설계가 되었다.

또한 저자들은 데이터 규모가 충분히 크기 때문에 과적합 위험이 낮다고 보고, 학습 파이프라인을 크게 단순화했다. 인코더를 ImageNet 가중치로 초기화하지 않고 **처음부터 학습**하며, projection도 비선형 대신 **linear projection**만 사용하고, 데이터 증강은 **random square crop 하나만** 적용한다.

<br/>

### 모델 아키텍처

**Image Encoder**로는 두 계열을 실험한다.

- **ResNet 계열**: ResNet-50을 기반으로 ResNet-D 개선, anti-aliased rect-2 blur pooling을 적용하고, global average pooling을 **attention pooling**으로 대체했다. 크기는 EfficientNet 스타일로 너비·깊이·해상도를 함께 키워 RN50x4, RN50x16, RN50x64까지 확장한다.
- **ViT 계열**: Vision Transformer를 거의 그대로 사용하되, patch embedding과 positional embedding을 합친 직후에 layer normalization을 추가했다. ViT-B/32, ViT-B/16, ViT-L/14를 학습한다.

**Text Encoder**는 63M 파라미터 규모의 Transformer로, 12개 레이어와 512 차원, 8개 attention head로 구성된다. 49,152개 vocabulary의 lower-cased BPE를 사용하고, 시퀀스 길이는 연산 효율을 위해 **76 토큰**으로 제한한다. `[EOS]` 토큰 위치의 최종 레이어 activation을 문장 표현으로 사용하며, layer normalization 후 multi-modal 공간으로 projection한다.

학습은 32 epoch 동안 Adam(decoupled weight decay)과 cosine schedule로 진행되며, 배치 크기는 32,768이다. 가장 큰 ResNet인 RN50x64는 **592개의 V100 GPU로 18일**, ViT-L/14는 256개 V100으로 12일이 소요되었다. 최종적으로 가장 성능이 좋은 모델은 ViT-L/14를 336 픽셀 해상도로 1 epoch 추가 학습한 **ViT-L/14@336px**이다.

<br/>

### Zero-shot Transfer

<!--Figure: Figure 1 (b),(c) Create dataset classifier from label text / Use for zero-shot prediction-->

CLIP의 zero-shot 추론은 **텍스트 인코더를 이용해 분류기의 가중치를 즉석에서 생성**하는 방식으로 이루어진다.

1. 대상 데이터셋의 모든 클래스 이름을 텍스트 프롬프트로 만든다 (예: `"A photo of a dog."`)
2. 각 프롬프트를 텍스트 인코더에 통과시켜 임베딩을 얻는다. 이 임베딩들이 곧 **선형 분류기의 가중치 벡터** 역할을 한다.
3. 입력 이미지의 임베딩과 각 클래스 임베딩의 코사인 유사도를 계산하고, softmax를 취해 예측한다.

이 구조 덕분에 새로운 분류 문제에 대해 **어떠한 학습 데이터나 파라미터 업데이트도 없이** 클래스 이름만 바꿔주면 분류기가 만들어진다.

<br/>

#### Prompt Engineering과 Ensembling

클래스 이름만 그대로 넣으면 두 가지 문제가 발생한다. 하나는 **다의성(polysemy)** 으로, 예를 들어 ImageNet의 `crane`은 두루미와 크레인 중 어느 쪽인지 알 수 없고 Oxford-IIIT Pets의 `boxer`는 견종이 아니라 권투 선수로 해석될 수 있다. 다른 하나는 **분포 불일치**로, 학습 데이터의 텍스트는 대개 단어 하나가 아니라 완결된 문장이었으므로 단어만 입력하면 사전 학습 분포와 어긋난다.

이에 논문은 두 가지 기법을 사용한다.

- **Prompt Engineering**: `"A photo of a {label}."`이라는 템플릿을 사용하는 것만으로 ImageNet 정확도가 **1.3%** 향상된다. 나아가 태스크의 도메인을 명시하면(`"A photo of a {label}, a type of pet."`, 위성 사진의 경우 `"a satellite photo of a {label}."`) 추가적인 개선을 얻는다.
- **Prompt Ensembling**: 서로 다른 문맥의 프롬프트 **80개**를 만들어 각각의 텍스트 임베딩을 구한 뒤 이를 평균내어 하나의 분류기 가중치로 사용한다. 이 방식으로 ImageNet 정확도가 추가로 **3.5%** 향상된다.

두 기법을 합치면 ImageNet에서 약 **5%** 의 성능 향상을 얻는다. 이는 연산 비용을 전혀 늘리지 않고 얻는 개선이라는 점에서 의미가 크며, 이후 **프롬프트 설계 자체가 vision 모델의 성능 요소가 되는** 흐름을 만들었다.

<br/>

## 실험 결과

### Zero-shot 성능

<!--Figure: Figure 5 (Zero-shot CLIP vs Linear Probe on ResNet-50)-->

- **기존 zero-shot 연구 대비 압도적 개선**: 이전의 대표적 zero-shot 전이 연구인 Visual N-Grams는 ImageNet에서 11.5%의 정확도를 기록했으나, CLIP은 동일한 설정에서 **76.2%** 를 달성하며 원본 supervised ResNet-50과 동등한 수준에 도달했다. 학습 이미지는 한 장도 사용하지 않았다.
- **supervised baseline과의 비교**: zero-shot CLIP을 ResNet-50 feature 위에 학습한 **fully supervised linear classifier**와 27개 데이터셋에서 비교했을 때, CLIP이 **16개 데이터셋에서 우위**를 보였다. 특히 일반적인 객체 인식(STL-10, CIFAR10, Food101 등)과 동영상 행동 인식(Kinetics700, UCF101)에서 강했는데, 자연어 supervision이 객체 중심의 라벨보다 **동사와 행위를 포함한 넓은 시각적 개념**을 다룰 수 있기 때문으로 해석된다.
- **취약한 영역**: 반면 위성 이미지 분류(EuroSAT), 림프절 종양 검출(PatchCamelyon), 자율주행 관련 거리 추정(KITTI), 합성 장면에서의 객체 개수 세기(CLEVRCounts) 등 **전문적이거나 추상적인 태스크**에서는 성능이 크게 떨어졌다.

<br/>

### Few-shot 및 Representation Learning

<!--Figure: Figure 6 (Zero-shot CLIP vs few-shot linear probes)-->

- **zero-shot ≈ 4-shot**: zero-shot CLIP은 놀랍게도 **동일한 CLIP feature 위에 4-shot으로 학습한 linear probe와 평균적으로 대등한** 성능을 보인다. 이는 자연어 프롬프트가 "개념을 직접 지정"하는 반면 few-shot linear probe는 소수의 예시로부터 개념을 **간접적으로 추론**해야 하기 때문이며, 예시가 적을수록 이 간접성의 손해가 크다는 것을 시사한다. 실제로 CLIP feature의 linear probe가 zero-shot을 넘어서려면 클래스당 약 4개 이상의 예시가 필요했다.
- **표현 학습 자체의 품질**: linear probe 평가에서 최고 성능 CLIP 모델은 27개 데이터셋 평균에서 기존 최고 수준 모델인 **Noisy Student EfficientNet-L2를 능가**했다. 즉 CLIP은 zero-shot 능력뿐 아니라 **feature extractor로서의 품질** 자체도 최상위권이다.
- **연산량에 따른 scaling**: 모델 크기를 44배 범위에서 변화시키며 측정한 결과, zero-shot 성능은 연산량에 대해 대체로 **부드러운 log-linear scaling**을 보였다. 다만 개별 데이터셋 단위로 보면 편차가 상당히 컸다.

<br/>

### Distribution Shift에 대한 강건성

<!--Figure: Figure 13 (Effective robustness on natural distribution shift)-->

기존 ImageNet 모델들은 ImageNet 정확도가 높아져도 **자연적 분포 변화(natural distribution shift)** 하에서의 정확도는 그만큼 따라오지 못한다는 문제가 알려져 있었다. ImageNetV2, ImageNet-R, ObjectNet, ImageNet-Sketch, ImageNet-A 등이 그 평가 대상이다.

zero-shot CLIP은 이 격차를 크게 줄여, 기존 모델 대비 **effective robustness를 최대 75%까지 개선**했다. ImageNet 자체의 정확도가 동등한 ResNet-101과 비교했을 때, 분포가 이동한 데이터셋에서는 CLIP이 훨씬 높은 정확도를 유지한다.

흥미로운 점은 CLIP을 ImageNet에 fine-tuning하면 ImageNet 정확도는 올라가지만 **강건성은 오히려 떨어진다**는 것이다. 이는 강건성의 원천이 모델 구조나 규모가 아니라, **특정 데이터셋 분포에 맞춰 학습되지 않았다는 사실 자체**에 있음을 보여준다.

<br/>

## 한계점

- **평균 수준에 머무르는 zero-shot 절대 성능**: zero-shot CLIP은 ResNet-50과 동등한 수준으로, 대부분의 데이터셋에서 각 태스크의 SOTA에는 크게 못 미친다. 저자들은 현재의 학습 방식으로 전반적인 SOTA에 도달하려면 **약 1000배의 연산량**이 추가로 필요할 것으로 추정하며, 이는 현재 하드웨어로는 실현 불가능한 수준이라고 밝힌다.
- **fine-grained·추상적·체계적 태스크의 취약성**: 자동차 모델, 꽃 품종, 항공기 기종 구분과 같은 **세분화된 분류**와, 이미지 내 객체 개수 세기나 가장 가까운 자동차까지의 거리 추정처럼 **추상적·체계적 추론**을 요구하는 태스크에서는 성능이 무작위에 가까운 경우도 있다.
- **진정한 out-of-distribution 데이터에서의 붕괴**: CLIP은 자연 이미지의 분포 변화에는 강건하지만, 사전 학습 분포에서 완전히 벗어난 데이터에는 여전히 취약하다. 대표적으로 **손글씨 숫자 MNIST에서 88%** 에 그치는데, 이는 원시 픽셀에 로지스틱 회귀를 적용한 단순 baseline보다도 낮은 수치이다. 4억 쌍의 데이터에도 MNIST와 유사한 이미지가 거의 없었기 때문으로, **CLIP이 일반화 문제를 근본적으로 해결한 것은 아님**을 드러낸다.
- **주어진 후보 중에서만 선택 가능**: CLIP의 zero-shot 분류기는 사람이 제공한 클래스 이름 집합 안에서만 예측할 수 있으며, 새로운 개념을 **스스로 생성해내지 못한다**. 저자들은 image captioning처럼 생성적 능력을 결합하는 방향을 후속 과제로 제시한다.
- **낮은 데이터 효율성**: 32 epoch 동안 4억 쌍을 학습하므로 모델은 총 **128억 장의 이미지**를 보게 된다. 사람이 하루에 한 장씩 본다고 가정하면 3500만 년에 해당하는 양으로, 데이터 효율성 측면에서는 매우 비효율적이다.
- **평가 방법론의 엄밀성 문제**: 프롬프트 설계와 하이퍼파라미터 선택 과정에서 평가용 테스트셋을 반복적으로 참조했기 때문에, 엄밀한 의미의 zero-shot 평가라고 보기 어렵다. 또한 평가에 사용된 데이터셋들 역시 기존 비전 연구의 관습에 따라 선택된 것이어서, CLIP의 능력을 측정하기 위해 설계된 벤치마크가 아니다.
- **사회적 편향과 오용 가능성**: 필터링되지 않은 웹 데이터로 학습되었기 때문에 사회적 편향을 그대로 학습한다. FairFace 기반 평가에서 특정 인종·연령 집단의 이미지를 범죄 관련 카테고리나 비인간 카테고리로 오분류하는 경향이 확인되었으며, 클래스 이름 설계에 따라 결과가 크게 달라졌다. 또한 임의의 카테고리를 즉석에서 정의할 수 있다는 CLIP의 강점은 **감시(surveillance) 용도로의 전용**을 쉽게 만든다는 위험도 함께 갖는다.

<br/>

## 인사이트

- **supervision의 형태가 확장성을 결정한다**: CLIP의 핵심 기여는 새로운 아키텍처가 아니라 **감독 신호를 고정 라벨에서 자연어로 바꾼 것**이다. 라벨링 비용이라는 병목을 제거하자 데이터 규모가 수백 배 커졌고, 그 결과로 zero-shot 전이 능력이 따라왔다. "무엇을 학습시킬 것인가"만큼 "어떤 형태의 감독을 쓸 것인가"가 중요하다는 점을 보여준 사례이다.
- **목적함수의 난이도를 낮추는 것이 곧 scale-up의 조건**: 캡션을 정확히 생성하는 어려운 문제를 짝을 맞히는 쉬운 문제로 완화한 것이 12배의 효율 개선을 낳았다. 대규모 학습에서는 목적함수의 정교함보다 **단위 연산당 학습 신호의 효율**이 더 결정적일 수 있음을 시사한다.
- **공유 임베딩 공간이 범용 인터페이스가 되다**: 이미지와 텍스트를 같은 공간에 정렬시킨다는 아이디어는 분류를 넘어 광범위하게 재사용되었다. **DALL·E 2**의 이미지 prior, **Stable Diffusion**의 텍스트 조건화, **BLIP-2·LLaVA** 등 대부분의 VLM이 채택한 vision encoder, open-vocabulary detection·segmentation이 모두 CLIP 임베딩 위에서 출발한다. CLIP은 하나의 모델이라기보다 **이후 멀티모달 연구의 공통 기반 부품**이 되었다.
- **"재학습 없이 텍스트로 태스크를 지정한다"는 패러다임**: 학습이 아니라 프롬프트로 태스크를 정의하는 방식은 NLP에서 GPT-3가 보여준 흐름을 vision으로 옮겨온 것이며, 두 분야가 **foundation model이라는 공통 개념으로 수렴**하는 전환점이 되었다.
- **강건성은 규모가 아니라 학습 분포에서 온다**: fine-tuning이 정확도를 올리면서 오히려 강건성을 떨어뜨린 관찰은, 벤치마크 정확도와 실제 배포 환경에서의 신뢰성이 별개의 축임을 보여준다. 이는 이후 모델 평가에서 **분포 변화에 대한 강건성을 독립적인 지표로 다루는** 관행에 영향을 주었다.
- **재현과 개선으로 이어진 확장**: 학습 데이터가 비공개였던 한계는 **LAION-400M/5B**와 **OpenCLIP**의 오픈소스 재현으로 보완되었고, 목적함수 측면에서는 softmax 대신 sigmoid 손실을 사용해 대규모 배치 의존성을 낮춘 **SigLIP** 등으로 발전이 이어졌다.

<br/>
